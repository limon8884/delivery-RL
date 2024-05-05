import typing
import torch
import json
import logging
import warnings
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

from src.objects import (
    Gamble,
    Assignment,
)
from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.dispatchs.base_dispatch import BaseDispatch
from src.dispatchs.hungarian_dispatch import HungarianDispatch
from src.dispatchs.greedy_dispatch import GreedyDispatch
from src.dispatchs.scorers import DistanceScorer
from src.router_makers import AppendRouteMaker
from src.networks.encoders import GambleEncoder
from src.networks.claim_attention import ClaimAttention
from src.utils import compulte_claims_to_couriers_distances

from src.reinforcement.base import (
    Action,
    State,
    BaseEnvironment,
    BaseActorCritic,
    Runner,
    GAE,
    Buffer,
    PPO,
    CloningPPO,
    Trajectory,
    MetricLogger,
    TrajectorySampler,
    RewardNormalizer,
    InferenceMetricsRunner,
    make_optimizer,
    BaseMaker,
)


# logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.DEBUG)
# LOGGER = logging.getLogger(__name__)

PAD_MASK_VALUE = -1e9
FAKE_MASK_VALUE = -1e7
PREV_CHOICE_MASK_VALUE = -1e9
FULL_ORDER_MASK_VALUE = -1e9


class DeliveryAction(Action):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def to_index(self) -> int:
        return self.idx


class DeliveryState(State):
    def __init__(self, claim_embs: np.ndarray, couriers_embs: typing.Optional[np.ndarray],
                 orders_embs: typing.Optional[np.ndarray], prev_idxs: list[int], orders_full_masks: list[bool],
                 claim_to_couries_dists: np.ndarray, gamble_features: np.ndarray, claim_idx: int) -> None:
        self.claim_embs = claim_embs
        self.couriers_embs = couriers_embs
        self.orders_embs = orders_embs
        self.prev_idxs = prev_idxs
        self.claim_to_couries_dists = claim_to_couries_dists
        self.orders_full_masks = orders_full_masks
        self.gamble_features = gamble_features
        self.claim_idx = claim_idx

    def last(self) -> int:
        len_crr = len(self.couriers_embs) if self.couriers_embs is not None else 0
        len_ord = len(self.orders_embs) if self.orders_embs is not None else 0
        return len_crr + len_ord

    def greedy(self) -> int:
        mask = np.zeros(self.last() + 1)
        mask[-1] = 1e9
        mask[np.array(self.prev_idxs, dtype=np.int32)] = 1e9
        idx = np.argmin(self.claim_to_couries_dists + mask)
        return idx

    def is_courier(self, idx: int) -> bool:
        len_crr = len(self.couriers_embs) if self.couriers_embs is not None else 0
        return idx < len_crr

    def has_free_couriers(self) -> bool:
        if self.couriers_embs is None:
            return False
        prev_idxs = set(self.prev_idxs)
        all_free_couriers = set(range(len(self.couriers_embs)))
        return len(all_free_couriers - prev_idxs) > 0

    def __hash__(self) -> int:
        return hash(
            hash(tuple(map(tuple, self.claim_embs))) +
            (hash(tuple(map(tuple, self.couriers_embs))) if self.couriers_embs is not None else 0) +
            (hash(tuple(map(tuple, self.orders_embs))) if self.orders_embs is not None else 0)
        )


class DeliveryRewarder:
    def __init__(self, **kwargs) -> None:
        self.coef_reward_completed = kwargs['coef_reward_completed']
        self.coef_reward_assigned = kwargs['coef_reward_assigned']
        self.coef_reward_cancelled = kwargs['coef_reward_cancelled']
        self.coef_reward_distance = kwargs['coef_reward_distance']
        self.coef_reward_prohibited = kwargs['coef_reward_prohibited']

    def __call__(self, assignment_statistics: dict[str, float]) -> float:
        completed = assignment_statistics['completed_claims']
        assigned = assignment_statistics['assigned_not_batched_claims'] \
            + assignment_statistics['assigned_batched_claims']
        cancelled = assignment_statistics['cancelled_claims']
        distance = assignment_statistics['assigned_not_batched_orders_arrival_distance']
        prohibited = assignment_statistics['prohibited_assignments']
        return self.coef_reward_completed * completed \
            + self.coef_reward_assigned * assigned \
            - self.coef_reward_cancelled * cancelled \
            - self.coef_reward_distance * distance \
            - self.coef_reward_prohibited * prohibited


class DeliveryEnvironment(BaseEnvironment):
    def __init__(self, simulator: Simulator, rewarder: DeliveryRewarder, **kwargs) -> None:
        self.kwargs = kwargs
        # self.max_num_points_in_route = kwargs['max_num_points_in_route']
        # self.use_dist = kwargs['use_dist']
        # self.use_route = kwargs['use_route']
        self.num_gambles = kwargs['num_gambles_in_day']
        self.simulator = simulator
        self.device = kwargs['device']
        self.rewarder = rewarder

    def copy(self) -> 'DeliveryEnvironment':
        return DeliveryEnvironment(
            simulator=deepcopy(self.simulator),
            rewarder=self.rewarder,
            **self.kwargs
        )

    def reset(self, seed: int | None = None) -> DeliveryState:
        self._iter = 0
        self._claim_idx: int = 0
        self._prev_idxs: list[int] = []
        self._assignments: Assignment = Assignment([])
        self._assignment_statistics: dict[str, float] = {}
        self.simulator.reset()
        self._update_next_gamble()
        state = self._make_state_from_gamble_dict()
        return state

    def step(self, action: DeliveryAction) -> tuple[DeliveryState, float, bool, dict[str, float]]:
        if self.__getattribute__('_iter') is None:
            raise RuntimeError('Call reset before doing steps')
        # LOGGER.debug(f'fake assignment: {action.to_index() == len(self._gamble.orders) + len(self._gamble.couriers)}')
        self._update_assignments(action)
        reward = 0.0
        done = False
        info = {}
        if self._claim_idx == len(self.embs_dict['clm']):
            self._update_next_gamble()
            reward = self.rewarder(self._assignment_statistics)
            info = self._assignment_statistics
        new_state = self._make_state_from_gamble_dict()
        if self._iter == self.num_gambles:
            done = True
            self._iter = 0
        return new_state, reward, done, info

    def _update_next_gamble(self):
        self.simulator.next(self._assignments)
        self._statistics_update(self.simulator.assignment_statistics)
        self._gamble = self.simulator.get_state()
        while len(self._gamble.claims) == 0:
            self.simulator.next(Assignment([]))
            self._statistics_add(self.simulator.assignment_statistics)
            self._gamble = self.simulator.get_state()
            self._iter += 1
        self.embs_dict = {
            'crr': np.stack([crr.to_numpy(**self.kwargs) for crr in self._gamble.couriers], axis=0)
            if len(self._gamble.couriers)
            else None,
            'clm': np.stack([clm.to_numpy(**self.kwargs) for clm in self._gamble.claims], axis=0),
            'ord': np.stack([ord.to_numpy(**self.kwargs) for ord in self._gamble.orders], axis=0)
            if len(self._gamble.orders) > 0
            else None,
            'gmb': self._gamble.to_numpy(**self.kwargs),
            'ord_masks': [ord.has_full_route(max_num_points_in_route=self.kwargs['max_num_points_in_route'])
                          for ord in self._gamble.orders],
            'dists': compulte_claims_to_couriers_distances(self._gamble,
                                                           distance_norm_constant=self.kwargs['distance_norm_constant'])
        }
        self._assignments = Assignment([])
        self._claim_idx = 0
        self._prev_idxs = []
        self._iter += 1

    def _update_assignments(self, action: DeliveryAction):
        if action.idx < len(self._gamble.couriers):
            self._assignments.ids.append((
                self._gamble.couriers[action.idx].id,
                self._gamble.claims[self._claim_idx].id
            ))
            self._prev_idxs.append(action.idx)
        elif action.idx - len(self._gamble.couriers) < len(self._gamble.orders):
            self._assignments.ids.append((
                self._gamble.orders[action.idx - len(self._gamble.couriers)].courier.id,
                self._gamble.claims[self._claim_idx].id
            ))
            self._prev_idxs.append(action.idx)
        elif action.idx > len(self._gamble.couriers) + len(self._gamble.orders):
            raise RuntimeError('Invalid action!')
        self._claim_idx += 1

    def _make_state_from_gamble_dict(self) -> DeliveryState:
        claim_to_couries_dists = self.embs_dict['dists'][self._claim_idx]
        return DeliveryState(
            claim_embs=self.embs_dict['clm'],
            claim_to_couries_dists=claim_to_couries_dists,
            couriers_embs=self.embs_dict['crr'],
            orders_embs=self.embs_dict['ord'],
            gamble_features=self.embs_dict['gmb'],
            prev_idxs=self._prev_idxs.copy(),
            orders_full_masks=self.embs_dict['ord_masks'],
            claim_idx=self._claim_idx,
        )

    def _statistics_add(self, new_stats: dict[str, float]) -> None:
        for k, v in new_stats.items():
            if k not in self._assignment_statistics:
                self._assignment_statistics[k] = 0.0
            self._assignment_statistics[k] += v
        if 'num steps' not in self._assignment_statistics:
            self._assignment_statistics['num steps'] = 0
        self._assignment_statistics['num steps'] += 1

    def _statistics_update(self, new_stats: dict[str, float]) -> None:
        self._assignment_statistics = {}
        self._statistics_add(new_stats)


class DeliveryActorCritic(BaseActorCritic):
    def __init__(self,
                 gamble_encoder: GambleEncoder,
                 attention: typing.Optional[nn.Transformer],
                 clm_emb_size: int,
                 co_emb_size: int,
                 gmb_emb_size: int,
                 **kwargs) -> None:
        super().__init__()
        self.gamble_encoder = gamble_encoder
        assert attention is not None, 'Not implemented yet'
        self.attention = attention
        # self.claim_attention = claim_attention
        self.clm_add_emb_size = clm_emb_size + 2
        self.co_add_emb_size = co_emb_size + gmb_emb_size + clm_emb_size + 2
        self.temperature = kwargs['exploration_temperature']
        self.device = kwargs['device']
        # self.mask_fake_crr = kwargs['mask_fake_crr']
        self.use_dist = kwargs['use_dist']
        # self.use_masks = kwargs['use_masks']

        self.clm_adaptor = nn.Linear(self.clm_add_emb_size, self.attention.d_model).to(self.device)
        self.co_adaptor = nn.Linear(self.co_add_emb_size, self.attention.d_model).to(self.device)
        self.policy_head = nn.Sequential(
            nn.Linear(self.attention.d_model, self.attention.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.attention.d_model, self.attention.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.attention.d_model, 1),
        ).to(self.device)
        self.value_head = nn.Sequential(
            nn.Linear(self.attention.d_model, self.attention.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.attention.d_model, self.attention.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.attention.d_model, 1),
        ).to(self.device)

    def forward(self, state_list: list[DeliveryState]) -> None:
        policy_tens, val_tens = self._make_policy_value_tensors(state_list)
        assert not torch.isnan(policy_tens).any()
        self.log_probs = nn.functional.log_softmax(policy_tens / self.temperature, dim=-1)
        self.values = val_tens
        self._actions: typing.Optional[torch.Tensor] = None

    # def _make_padded_policy_value_tensors(self, states: list[DeliveryState]
    #                                       ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    #     policy_tens_list, value_tens_list = [], []
    #     for state in states:
    #         co_embs, claim_emb, gamble_features_emb = self._make_embeddings_tensors_from_state(state)
    #         additional_features = self._make_additional_features_from_state(state)
    #         coc_embs = torch.cat([co_embs, claim_emb.repeat(len(co_embs), 1),
    #                               gamble_features_emb.repeat(len(co_embs), 1),
    #                               additional_features], dim=-1)
    #         masks = self._make_masks(state)

    #         policy_tens = self.policy_head(coc_embs).squeeze(-1)
    #         if self.use_masks:
    #             policy_tens[masks] = FULL_ORDER_MASK_VALUE
    #         if self.mask_fake_crr:
    #             policy_tens[-1] = FAKE_MASK_VALUE
    #         policy_tens_list.append(policy_tens)

    #         value_tens = self.value_head(coc_embs).squeeze(-1)
    #         if self.use_masks:
    #             value_tens[masks] = 0.0
    #         if self.mask_fake_crr:
    #             value_tens[-1] = 0.0
    #         value_tens_list.append(value_tens.mean())

    #     policy_tens_result = pad_sequence(policy_tens_list, batch_first=True, padding_value=PAD_MASK_VALUE)
    #     value_tens_result = torch.tensor(value_tens_list, device=self.device, dtype=torch.float)
    #     return policy_tens_result, value_tens_result

    # def _make_masks(self, state: DeliveryState) -> torch.Tensor:
    #     # order full mask
    #     couriers_part = [False] * len(state.couriers_embs) if state.couriers_embs is not None else []
    #     orders_part = state.orders_full_masks
    #     order_full_mask = torch.tensor(couriers_part + orders_part + [False], device=self.device, dtype=torch.bool)
    #     # prev assigned
    #     prev_assigned_mask = torch.zeros(len(order_full_mask), dtype=torch.bool, device=self.device)
    #     prev_assigned_mask[state.prev_idxs] = True

    #     mask = prev_assigned_mask | order_full_mask
    #     return mask

    # def _make_additional_features_from_state(self, state: DeliveryState) -> torch.Tensor:
    #     num_crr_ord_fake = (len(state.couriers_embs) if state.couriers_embs is not None else 0) + \
    #         (len(state.orders_embs) if state.orders_embs is not None else 0) + 1
    #     num_clm = len(state.claim_embs)

    #     prev_idxs = torch.tensor(state.prev_idxs, dtype=torch.int64, device=self.device)
    #     prev_assigs = torch.zeros(size=(num_crr_ord_fake, 1), dtype=torch.float, device=self.device)
    #     prev_assigs[prev_idxs] = 1.0

    #     claim_idx = torch.ones(size=(num_crr_ord_fake, 1), dtype=torch.float,
    #                            device=self.device) * state.claim_idx / num_clm

    #     if self.use_dist:
    #         dists = torch.tensor(state.claim_to_couries_dists, dtype=torch.float,
    #                              device=self.device).unsqueeze(-1)
    #         return torch.cat([prev_assigs, claim_idx, dists], dim=-1)
    #     return torch.cat([prev_assigs, claim_idx], dim=-1)

    def _make_policy_value_tensors(self, states: list[DeliveryState]) -> tuple[torch.Tensor, torch.Tensor]:
        clm_embs_list, co_embs_list = [], []
        for state in states:
            clm_embs, co_embs = self._make_clm_co_tensors(state)
            clm_embs_list.append(clm_embs)
            co_embs_list.append(co_embs)
        clm_embs = pad_sequence(clm_embs_list, batch_first=True, padding_value=0.0)
        co_embs = pad_sequence(co_embs_list, batch_first=True, padding_value=0.0)
        clm_masks = self._make_mask_from_lengths([len(e) for e in clm_embs_list], self.device)
        co_masks = self._make_mask_from_lengths([len(e) for e in co_embs_list], self.device)
        attn = self.attention(
            src=clm_embs,
            tgt=co_embs,
            src_key_padding_mask=clm_masks,
            tgt_key_padding_mask=co_masks,
            memory_key_padding_mask=clm_masks
        )
        policy_tens = self.policy_head(attn).squeeze(-1)  # (bs, max_seq_len)
        value_tens = self.value_head(attn).squeeze(-1)  # (bs, max_seq_len)
        policy = torch.where(co_masks, PAD_MASK_VALUE, policy_tens)
        value = torch.where(co_masks, 0.0, value_tens).mean(-1)
        return policy, value

    def _make_clm_co_tensors(self, state: DeliveryState) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns 2 tensors:
        * clm_embs: (bs, max_num_clm, clm_add_emb_size)
        * co_embs: (bs, max_num_crr_ord_fake, co_add_emb_size)
        '''
        embs_dict = {
            'clm': state.claim_embs,
            'crr': state.couriers_embs,
            'ord': state.orders_embs,
            'gmb': state.gamble_features.reshape(1, -1),
        }
        encoded_dict = self.gamble_encoder(embs_dict)

        fake_crr = torch.ones(size=(1, self.gamble_encoder.courier_encoder.item_embedding_dim), device=self.device)
        co_embs = torch.cat(
            ([encoded_dict['crr']] if encoded_dict['crr'] is not None else []) +
            ([encoded_dict['ord']] if encoded_dict['ord'] is not None else []) +
            [fake_crr], dim=0)

        assert encoded_dict['gmb'] is not None
        gmb_emb = encoded_dict['gmb'].repeat(len(co_embs), 1)

        prev_idxs = torch.tensor(state.prev_idxs, dtype=torch.int64, device=self.device)
        prev_assigs = torch.zeros(size=(len(co_embs),), dtype=torch.float, device=self.device)
        prev_assigs[prev_idxs] = 1.0

        assert encoded_dict['clm'] is not None
        clm_emb = encoded_dict['clm'][state.claim_idx].unsqueeze(0).repeat(len(co_embs), 1)
        if self.use_dist:
            dists = torch.tensor(state.claim_to_couries_dists, dtype=torch.float,
                                 device=self.device).unsqueeze(-1)
        else:
            dists = torch.zeros((len(co_embs), 1)).to(self.device)
        assert co_embs.ndim == 2
        assert gmb_emb.ndim == 2
        assert prev_assigs.ndim == 1
        assert clm_emb.ndim == 2
        assert dists.ndim == 2
        co_add_emb = torch.cat([co_embs, gmb_emb, prev_assigs.unsqueeze(-1), clm_emb, dists], dim=-1)
        co_final_embs = self.co_adaptor(co_add_emb)

        clm_embs = encoded_dict['clm']
        clm_prev_assigs = torch.where(torch.arange(len(clm_embs)).to(self.device) < state.claim_idx, 1.0, 0.0)
        clm_idx = torch.zeros(len(clm_embs)).to(self.device)
        clm_idx[state.claim_idx] = 1.0
        clm_add_emb = torch.cat([clm_embs, clm_prev_assigs.unsqueeze(-1), clm_idx.unsqueeze(-1)], dim=-1)
        clm_final_embs = self.clm_adaptor(clm_add_emb)

        return clm_final_embs, co_final_embs

    @staticmethod
    def _make_mask_from_lengths(lengths: list[int], device: str) -> torch.Tensor:
        max_len = max(lengths)
        lengths_tens = torch.tensor(lengths).unsqueeze(-1).to(device)
        arange_tens = torch.arange(max_len).expand(len(lengths), max_len).to(device)
        return arange_tens >= lengths_tens

    # def _make_embeddings_tensors_from_state(self, state: DeliveryState
    #                                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     '''
    #     Returns 3 tensors:
    #     * co_embs: (n_crr + n_ord + 1, co_emb_size)
    #     * clm_emb: (n_clm, clm_emb_size)
    #     * gmb_emb: (1, gmb_emb_size)
    #     '''
    #     # claim_embs = state.claim_embs if self.claim_attention else state.claim_embs[state.claim_idx][None, :]
    #     embs_dict = {
    #         'clm': state.claim_embs,
    #         'crr': state.couriers_embs,
    #         'ord': state.orders_embs,
    #         'gmb': state.gamble_features.reshape(1, -1),
    #     }
    #     encoded_dict = self.gamble_encoder(embs_dict)
    #     fake_crr = torch.ones(size=(1, self.gamble_encoder.courier_encoder.item_embedding_dim), device=self.device)
    #     co_embs = torch.cat(
    #         ([encoded_dict['crr']] if encoded_dict['crr'] is not None else []) +
    #         ([encoded_dict['ord']] if encoded_dict['ord'] is not None else []) +
    #         [fake_crr], dim=0)
    #     assert encoded_dict['clm'] is not None and encoded_dict['gmb'] is not None
    #     gmb_emb = encoded_dict['gmb']
    #     if self.claim_attention is not None:
    #         clm_emb = self.claim_attention(encoded_dict['clm'], state.claim_idx)
    #     else:
    #         clm_emb = encoded_dict['clm']
    #     return co_embs, clm_emb, gmb_emb

    def get_actions_list(self, best_actions=False) -> list[Action]:
        if best_actions:
            self._actions = torch.argmax(self.log_probs, dim=-1)
        else:
            self._actions = torch.distributions.categorical.Categorical(logits=self.log_probs).sample()
        assert self._actions is not None
        return [DeliveryAction(a) for a in self._actions.tolist()]

    def get_log_probs_list(self) -> list[float]:
        assert self._actions is not None, 'call `get_actions_list` before'
        return self.log_probs[torch.arange(len(self._actions), device=self.device), self._actions].tolist()

    def get_values_list(self) -> list[float]:
        return self.values.tolist()

    def get_log_probs_tensor(self) -> torch.Tensor:
        return self.log_probs

    def get_values_tensor(self) -> torch.Tensor:
        return self.values


class DeliveryInferenceMetricsRunner(InferenceMetricsRunner):
    @staticmethod
    def get_metrics_from_trajectory(trajs: list[Trajectory]) -> dict[str, float]:
        cumulative_metrics: dict[str, float] = defaultdict(float)
        total_iters = 0
        for traj in trajs:
            for reward, log_prob_chosen, entropy, action, state in zip(
                    traj.rewards, traj.log_probs_chosen, traj.entropies, traj.actions, traj.states):
                assert isinstance(state, DeliveryState)
                total_iters += 1
                cumulative_metrics['step reward'] += reward
                cumulative_metrics['chosen prob'] += np.exp(log_prob_chosen)
                cumulative_metrics['entropy'] += entropy
                cumulative_metrics['has available couriers'] += int(state.has_free_couriers())

                # has available couriers and not assigned
                cumulative_metrics['not assigned'] += int(
                    action.to_index() == state.last() and state.has_free_couriers())
                # cumulative_metrics['invalid actions'] += int(action.to_index() > state.last())

                # has available couriers and greedy assigned
                cumulative_metrics['greedy'] += int(
                    action.to_index() == state.greedy() and state.has_free_couriers())
        results = {('PPO: ' + metric): (value / total_iters) for metric, value in cumulative_metrics.items()}
        results.update({
            'not assigned | has available': cumulative_metrics['not assigned'] / cumulative_metrics['has available couriers'],
            'greedy | has available': cumulative_metrics['greedy'] / cumulative_metrics['has available couriers'],
        })
        return results


class DeliveryMaker(BaseMaker):
    def __init__(self, **kwargs) -> None:
        batch_size = kwargs['batch_size']
        max_num_points_in_route = kwargs['max_num_points_in_route']
        device = kwargs['device']

        simulator_config_path = Path(kwargs['simulator_cfg_path'])
        model_size = kwargs['model_size']
        network_config_path = Path(kwargs['network_cfg_path'])
        with open(network_config_path) as f:
            net_cfg = json.load(f)
            encoder_cfg = net_cfg['encoder'][model_size]
            attn_cfg = net_cfg['attention'][model_size]

        gamble_encoder = GambleEncoder(**kwargs, **encoder_cfg)
        # claim_attention = ClaimAttention(**kwargs, **attn_cfg) if kwargs['use_attn'] else None
        attention = nn.Transformer(**attn_cfg).to(device)
        reader = DataReader.from_config(config_path=simulator_config_path,
                                        sampler_mode=kwargs['sampler_mode'], db_logger=None)
        route_maker = AppendRouteMaker(max_points_lenght=max_num_points_in_route, cutoff_radius=0.0)
        sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=simulator_config_path, db_logger=None)
        rewarder = DeliveryRewarder(**kwargs)
        self._train_metric_logger = MetricLogger(use_wandb=kwargs['use_wandb'])
        self._env = DeliveryEnvironment(simulator=sim, rewarder=rewarder, **kwargs)
        self._ac = DeliveryActorCritic(gamble_encoder=gamble_encoder, attention=attention,
                                       clm_emb_size=encoder_cfg['claim_embedding_dim'],
                                       co_emb_size=encoder_cfg['courier_order_embedding_dim'],
                                       gmb_emb_size=encoder_cfg['gamble_features_embedding_dim'],
                                       **kwargs)
        if kwargs['load_checkpoint']:
            self._ac.load_state_dict(torch.load(kwargs['load_checkpoint'], map_location=device))
        opt = make_optimizer(self._ac.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=kwargs['scheduler_max_lr'],
                                                        total_steps=kwargs['total_iters'],
                                                        pct_start=kwargs['scheduler_pct_start'])
        self._ppo = PPO(
            actor_critic=self._ac,
            opt=opt,
            scheduler=scheduler,
            metric_logger=self._train_metric_logger,
            **kwargs
            )
        runner = Runner(environment=self._env, actor_critic=self._ac, n_envs=kwargs['n_envs'],
                        trajectory_length=kwargs['trajectory_length'])
        if kwargs['use_cloning'] == 'hungarian':
            self._ppo = CloningPPO(actor_critic=self._ac, opt=opt, scheduler=scheduler,
                                   metric_logger=self._train_metric_logger, **kwargs)
            runner = CloningDeliveryRunner(dispatch=HungarianDispatch(DistanceScorer()),
                                           simulator=sim, rewarder=rewarder, **kwargs)
        elif kwargs['use_cloning'] == 'greedy':
            self._ppo = CloningPPO(actor_critic=self._ac, opt=opt, scheduler=scheduler,
                                   metric_logger=self._train_metric_logger, **kwargs)
            runner = CloningDeliveryRunner(dispatch=GreedyDispatch(DistanceScorer()),
                                           simulator=sim, rewarder=rewarder, **kwargs)
        gae = GAE(gamma=kwargs['gae_gamma'], lambda_=kwargs['gae_lambda'])
        normalizer = RewardNormalizer(gamma=kwargs['reward_norm_gamma'], cliprange=kwargs['reward_norm_cliprange'])
        buffer = Buffer(gae, reward_normalizer=normalizer, device=device)
        self._sampler = TrajectorySampler(runner, buffer, num_epochs_per_traj=kwargs['num_epochs_per_traj'],
                                          batch_size=batch_size)

    @property
    def ppo(self) -> PPO:
        return self._ppo

    @property
    def sampler(self) -> TrajectorySampler:
        return self._sampler

    @property
    def actor_critic(self) -> BaseActorCritic:
        return self._ac

    @property
    def environment(self) -> BaseEnvironment:
        return self._env

    @property
    def metric_logger(self) -> MetricLogger:
        return self._train_metric_logger


class CloningDeliveryRunner:
    def __init__(self, dispatch: BaseDispatch, simulator: Simulator, rewarder: DeliveryRewarder, **kwargs) -> None:
        assert kwargs['n_envs'] == 1
        self.n_envs = 1
        self.dispatch = dispatch
        self.simulator: Simulator = deepcopy(simulator)
        self.rewarder = rewarder
        self.kwargs = kwargs
        # self.max_num_points_in_route = kwargs['max_num_points_in_route']
        # self.use_dist = kwargs['use_dist']
        # self.use_route = kwargs['use_route']
        self.num_gambles = kwargs['num_gambles_in_day']
        # self.device = kwargs['device']
        self.trajectory_length = kwargs['trajectory_length']

    def run(self) -> list[Trajectory]:
        state = self.reset()
        trajectory = Trajectory(state)
        for _ in range(self.trajectory_length):
            new_state, reward, done, info, action = self.step()
            trajectory.append(state, action, reward, done, 0.0, 0.0, 0.0)
            self._statistics.append(info)
            state = new_state
        trajectory.last_state = state
        trajectory.last_state_value = 0.0
        return [trajectory]

    def reset(self) -> DeliveryState:
        self._statistics = []
        self._iter = 0
        self._claim_idx: int = 0
        self._prev_idxs: list[int] = []
        self._assignments: Assignment = Assignment([])
        self._assignment_dict: dict[int, int] = {}
        self._assignment_statistics: dict[str, float] = {}
        self.simulator.reset()
        self._gamble = self.simulator.get_state()
        self._crr_id_to_index: dict[int, int] = {}
        self._update_next_gamble()
        return self._make_state_from_gamble_dict()

    def step(self) -> tuple[DeliveryState, float, bool, dict[str, float], DeliveryAction]:
        if self.__getattribute__('_iter') is None:
            raise RuntimeError('Call reset before doing steps')
        # self._update_assignments(action)
        action = self._make_action()
        self._claim_idx += 1
        reward = 0.0
        done = False
        info = {}
        if self._claim_idx == len(self.embs_dict['clm']):
            self._update_next_gamble()
            reward = self.rewarder(self._assignment_statistics)
            info = self._assignment_statistics
        new_state = self._make_state_from_gamble_dict()
        if self._iter == self.num_gambles:
            done = True
            self._iter = 0
        return new_state, reward, done, info, action

    def _make_action(self) -> DeliveryAction:
        clm_id = self._gamble.claims[self._claim_idx].id
        if clm_id in self._assignment_dict:
            crr_id = self._assignment_dict[clm_id]
        else:
            crr_id = -1
        act_index = self._crr_id_to_index[crr_id]
        self._prev_idxs.append(act_index)
        return DeliveryAction(act_index)

    def _make_crr_id_dict(self, gamble: Gamble) -> dict[int, int]:
        pad = 0
        d = {}
        for i, crr in enumerate(gamble.couriers):
            d[crr.id] = i + pad
        pad = len(gamble.couriers)
        for i, ord in enumerate(gamble.orders):
            d[ord.courier.id] = i + pad
        pad += len(gamble.orders)
        d[-1] = pad
        return d

    def _update_next_gamble(self):
        self.simulator.next(self._assignments)
        self._statistics_update(self.simulator.assignment_statistics)
        self._gamble = self.simulator.get_state()
        self._crr_id_to_index = self._make_crr_id_dict(self._gamble)
        self._assignments = self.dispatch(self._gamble)
        self._assignment_dict = {clm_id: crr_id for crr_id, clm_id in self._assignments.ids}
        while len(self._gamble.claims) == 0:
            self.simulator.next(Assignment([]))
            self._statistics_add(self.simulator.assignment_statistics)
            self._gamble = self.simulator.get_state()
            self._crr_id_to_index = self._make_crr_id_dict(self._gamble)
            self._assignments = self.dispatch(self._gamble)
            self._assignment_dict = {clm_id: crr_id for crr_id, clm_id in self._assignments.ids}
            self._iter += 1
        self.embs_dict = {
            'crr': np.stack([crr.to_numpy(**self.kwargs) for crr in self._gamble.couriers], axis=0)
            if len(self._gamble.couriers)
            else None,
            'clm': np.stack([clm.to_numpy(**self.kwargs) for clm in self._gamble.claims], axis=0),
            'ord': np.stack([ord.to_numpy(**self.kwargs) for ord in self._gamble.orders], axis=0)
            if len(self._gamble.orders) > 0
            else None,
            'gmb': self._gamble.to_numpy(**self.kwargs),
            'ord_masks': [ord.has_full_route(max_num_points_in_route=self.kwargs['max_num_points_in_route'])
                          for ord in self._gamble.orders],
            'dists': compulte_claims_to_couriers_distances(self._gamble,
                                                           distance_norm_constant=self.kwargs['distance_norm_constant'])
        }
        # self._assignments = Assignment([])
        self._claim_idx = 0
        self._prev_idxs = []
        self._iter += 1

    def _make_state_from_gamble_dict(self) -> DeliveryState:
        claim_to_couries_dists = self.embs_dict['dists'][self._claim_idx]
        return DeliveryState(
            claim_embs=self.embs_dict['clm'],
            claim_to_couries_dists=claim_to_couries_dists,
            couriers_embs=self.embs_dict['crr'],
            orders_embs=self.embs_dict['ord'],
            gamble_features=self.embs_dict['gmb'],
            prev_idxs=self._prev_idxs.copy(),
            orders_full_masks=self.embs_dict['ord_masks'],
            claim_idx=self._claim_idx,
        )

    def _statistics_add(self, new_stats: dict[str, float]) -> None:
        for k, v in new_stats.items():
            if k not in self._assignment_statistics:
                self._assignment_statistics[k] = 0.0
            self._assignment_statistics[k] += v
        if 'num steps' not in self._assignment_statistics:
            self._assignment_statistics['num steps'] = 0
        self._assignment_statistics['num steps'] += 1

    def _statistics_update(self, new_stats: dict[str, float]) -> None:
        self._assignment_statistics = {}
        self._statistics_add(new_stats)
