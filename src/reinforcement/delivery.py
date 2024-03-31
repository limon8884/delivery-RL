import typing
import torch
import json
import logging
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from copy import deepcopy

from src.objects import (
    Gamble,
    Assignment,
)
from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.router_makers import AppendRouteMaker
from src.networks.encoders import GambleEncoder


from src.reinforcement.base import (
    Action,
    State,
    BaseEnvironment,
    BaseActorCritic,
    Runner,
    GAE,
    Buffer,
    PPO,
    MetricLogger,
    TrajectorySampler,
    RewardNormalizer,
    # InferenceMetricsRunner,
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
    def __init__(self, claim_emb: np.ndarray, couriers_embs: typing.Optional[np.ndarray],
                 orders_embs: typing.Optional[np.ndarray], prev_idxs: list[int], orders_full_masks: list[bool]) -> None:
        self.claim_emb = claim_emb
        self.couriers_embs = couriers_embs
        self.orders_embs = orders_embs
        self.prev_idxs = prev_idxs
        self.orders_full_masks = orders_full_masks

    def size(self) -> int:
        len_crr = len(self.couriers_embs) if self.couriers_embs is not None else 0
        len_ord = len(self.orders_embs) if self.orders_embs is not None else 0
        return len_crr + len_ord

    def __hash__(self) -> int:
        return hash(
            hash(tuple(self.claim_emb)) +
            (hash(tuple(map(tuple, self.couriers_embs))) if self.couriers_embs is not None else 0) +
            (hash(tuple(map(tuple, self.orders_embs))) if self.orders_embs is not None else 0)
        )


class DeliveryRewarder:
    def __init__(self, **kwargs) -> None:
        self.coef_reward_completed = kwargs['coef_reward_completed']
        self.coef_reward_assigned = kwargs['coef_reward_assigned']
        self.coef_reward_cancelled = kwargs['coef_reward_cancelled']
        self.coef_reward_distance = kwargs['coef_reward_distance']

    def __call__(self, assignment_statistics: dict[str, float]) -> float:
        completed = assignment_statistics['completed_claims']
        assigned = assignment_statistics['assigned_not_batched_claims'] \
            + assignment_statistics['assigned_batched_claims']
        cancelled = assignment_statistics['cancelled_claims']
        distance = assignment_statistics['assigned_not_batched_orders_arrival_distance']
        return self.coef_reward_completed * completed \
            + self.coef_reward_assigned * assigned \
            - self.coef_reward_cancelled * cancelled \
            - self.coef_reward_distance * distance


class DeliveryEnvironment(BaseEnvironment):
    def __init__(self, simulator: Simulator, rewarder: DeliveryRewarder, **kwargs) -> None:
        self.max_num_points_in_route = kwargs['max_num_points_in_route']
        self.num_gambles = kwargs['num_gambles_in_day']
        self.simulator = simulator
        self.device = kwargs['device']
        self.rewarder = rewarder

    def copy(self) -> 'DeliveryEnvironment':
        return DeliveryEnvironment(
            simulator=deepcopy(self.simulator),
            rewarder=self.rewarder,
            max_num_points_in_route=self.max_num_points_in_route,
            num_gambles_in_day=self.num_gambles,
            device=self.device,
        )

    def reset(self) -> DeliveryState:
        self._iter = 0
        self._gamble: typing.Optional[Gamble] = None
        self._claim_idx: int = 0
        self._prev_idxs: list[int] = []
        self._assignments: Assignment = Assignment([])
        self._base_gamble_reward: float = 0.0
        self._assignment_statistics: dict[str, float] = {
            # 'free_couriers': 0.0,
            # 'free_claims': 0.0,
            # 'active_routes': 0.0,
        }
        self.simulator.reset()
        self._update_next_gamble()
        state = self._make_state_from_gamble_dict()
        return state

    def step(self, action: DeliveryAction) -> tuple[DeliveryState, float, bool, dict[str, float]]:
        if self.__getattribute__('_iter') is None:
            raise RuntimeError('Call reset before doing steps')
        # LOGGER.debug(f'fake assignment: {action.to_index() == len(self._gamble.orders) + len(self._gamble.couriers)}')
        self._update_assignments(action)
        reward = 0
        done = False
        info = {}
        if self._claim_idx == len(self.embs_dict['clm']):
            self._update_next_gamble()
            reward = self.rewarder(self._assignment_statistics)
            info = self._assignment_statistics
        new_state = self._make_state_from_gamble_dict()
        if self._iter == self.num_gambles:
            done = True
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
            'crr': np.stack([crr.to_numpy() for crr in self._gamble.couriers], axis=0)
            if len(self._gamble.couriers)
            else None,
            'clm': np.stack([clm.to_numpy() for clm in self._gamble.claims], axis=0),
            'ord': np.stack([ord.to_numpy(max_num_points_in_route=self.max_num_points_in_route)
                             for ord in self._gamble.orders], axis=0)
            if len(self._gamble.orders) > 0
            else None,
            'ord_masks': [ord.has_full_route(max_num_points_in_route=self.max_num_points_in_route)
                          for ord in self._gamble.orders]
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
        self._claim_idx += 1

    def _make_state_from_gamble_dict(self) -> DeliveryState:
        claim_emb = self.embs_dict['clm'][self._claim_idx]
        return DeliveryState(
            claim_emb=claim_emb,
            couriers_embs=self.embs_dict['crr'],
            orders_embs=self.embs_dict['ord'],
            prev_idxs=self._prev_idxs.copy(),
            orders_full_masks=self.embs_dict['ord_masks'],
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
        # self._assignment_statistics['free_couriers'] += new_stats['new_couriers'] \
        #     - new_stats['assigned_couriers'] \
        #     + new_stats['completed_orders'] \
        #     - new_stats['finished_couriers']
        # self._assignment_statistics['free_claims'] += new_stats['new_claims'] \
        #     - new_stats['assigned_not_batched_claims'] \
        #     - new_stats['assigned_batched_claims'] \
        #     - new_stats['cancelled_claims']
        # self._assignment_statistics['active_routes'] += new_stats['assigned_not_batched_claims'] \
        #     - new_stats['completed_orders']


class DeliveryActorCritic(BaseActorCritic):
    def __init__(self, gamble_encoder: GambleEncoder, clm_emb_size: int, temperature: float,
                 device, mask_fake_crr: bool = False) -> None:
        super().__init__()
        self.gamble_encoder = gamble_encoder
        self.clm_emb_size = clm_emb_size
        self.temperature = temperature
        self.mask_fake_crr = mask_fake_crr
        self.device = device

        # self.policy_matrix = nn.Parameter(torch.randn(clm_emb_size, clm_emb_size), requires_grad=True).to(device)
        # self.value_matrix = nn.Parameter(torch.randn(clm_emb_size, clm_emb_size), requires_grad=True).to(device)
        self.policy_head = nn.Sequential(
            nn.Linear(3 * clm_emb_size, clm_emb_size),
            nn.LeakyReLU(),
            nn.Linear(clm_emb_size, 1),
        ).to(device)
        self.value_head = nn.Sequential(
            nn.Linear(3 * clm_emb_size, clm_emb_size),
            nn.LeakyReLU(),
            nn.Linear(clm_emb_size, 1),
        ).to(device)

    def forward(self, state_list: list[DeliveryState]) -> None:
        policy_tens, val_tens = self._make_padded_policy_value_tensors(state_list)
        self.log_probs = nn.functional.log_softmax(policy_tens / self.temperature, dim=-1)
        self.values = val_tens
        self._actions = None

    def _make_padded_policy_value_tensors(self, states: list[DeliveryState]
                                          ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        policy_tens_list, value_tens_list = [], []
        for state in states:
            prev_idxs = torch.tensor(state.prev_idxs, dtype=torch.int64, device=self.device)
            co_embs, claim_emb = self._make_three_tensors_from_state(state)
            coc_embs = torch.cat([co_embs, claim_emb.repeat(len(co_embs), 1)], dim=-1)
            full_ord_masks = self._make_full_order_mask(state)

            policy_tens = self.policy_head(coc_embs).squeeze(-1)
            policy_tens[prev_idxs] = PREV_CHOICE_MASK_VALUE
            policy_tens[full_ord_masks] = FULL_ORDER_MASK_VALUE
            if self.mask_fake_crr:
                policy_tens[-1] = FAKE_MASK_VALUE
            policy_tens_list.append(policy_tens)

            value_tens = self.value_head(coc_embs).squeeze(-1)
            value_tens[prev_idxs] = 0.0
            value_tens[full_ord_masks] = 0.0
            if self.mask_fake_crr:
                value_tens[-1] = 0.0
            value_tens_list.append(value_tens.mean())

        policy_tens_result = pad_sequence(policy_tens_list, batch_first=True, padding_value=PAD_MASK_VALUE)
        value_tens_result = torch.tensor(value_tens_list, device=self.device)
        return policy_tens_result, value_tens_result

    def _make_full_order_mask(self, state: DeliveryState) -> torch.BoolTensor:
        couriers_part = [False] * len(state.couriers_embs) if state.couriers_embs is not None else []
        orders_part = state.orders_full_masks
        mask = torch.tensor(couriers_part + orders_part + [False], device=self.device, dtype=torch.bool)
        return mask

    def _make_three_tensors_from_state(self, state: DeliveryState
                                       ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        embs_dict = {
            'clm': state.claim_emb.reshape(1, -1),
            'crr': state.couriers_embs,
            'ord': state.orders_embs,
        }
        encoded_dict = self.gamble_encoder(embs_dict)
        fake_crr = torch.ones(size=(1, self.gamble_encoder.courier_encoder.item_embedding_dim), device=self.device)
        co_embs = torch.cat(
            ([encoded_dict['crr']] if encoded_dict['crr'] is not None else []) +
            ([encoded_dict['ord']] if encoded_dict['ord'] is not None else []) +
            [fake_crr], dim=0)
        return co_embs, encoded_dict['clm'][0]

    def get_actions_list(self, best_actions=False) -> list[Action]:
        if best_actions:
            self._actions = torch.argmax(self.log_probs, dim=-1)
        else:
            self._actions = torch.distributions.categorical.Categorical(logits=self.log_probs).sample()
        return [DeliveryAction(a) for a in self._actions.tolist()]

    def get_log_probs_list(self) -> list[float]:
        return self.log_probs[torch.arange(len(self._actions), device=self.device), self._actions].tolist()

    def get_values_list(self) -> list[float]:
        return self.values.tolist()

    def get_log_probs_tensor(self) -> torch.FloatTensor:
        return self.log_probs

    def get_values_tensor(self) -> torch.FloatTensor:
        return self.values


class DeliveryMaker(BaseMaker):
    def __init__(self, **kwargs) -> None:
        batch_size = kwargs['batch_size']
        max_num_points_in_route = kwargs['max_num_points_in_route']
        device = kwargs['device']

        simulator_config_path = Path(kwargs['simulator_cfg_path'])
        model_size = kwargs['model_size']
        network_config_path = Path(kwargs['network_cfg_path'])
        with open(network_config_path) as f:
            encoder_cfg = json.load(f)['encoder'][model_size]

        gamble_encoder = GambleEncoder(max_num_points_in_route=max_num_points_in_route, **encoder_cfg, device=device)
        reader = DataReader.from_config(config_path=simulator_config_path,
                                        sampler_mode=kwargs['sampler_mode'], db_logger=None)
        route_maker = AppendRouteMaker(max_points_lenght=max_num_points_in_route, cutoff_radius=0.0)
        sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=simulator_config_path, db_logger=None)
        rewarder = DeliveryRewarder(**kwargs)
        self._train_metric_logger = MetricLogger(use_wandb=kwargs['use_wandb'])
        self._env = DeliveryEnvironment(simulator=sim, rewarder=rewarder, **kwargs)
        self._ac = DeliveryActorCritic(gamble_encoder=gamble_encoder, clm_emb_size=encoder_cfg['claim_embedding_dim'],
                                       temperature=kwargs['exploration_temperature'],
                                       mask_fake_crr=kwargs['mask_fake_crr'], device=device)
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
                        trajectory_lenght=kwargs['trajectory_lenght'])
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
