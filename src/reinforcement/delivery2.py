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
from src.networks.backbones import TransformerBackbone
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
    MetricLogger,
    TrajectorySampler,
    RewardNormalizer,
    # InferenceMetricsRunner,
    make_optimizer,
    BaseMaker,
)


PAD_MASK_VALUE = -1e9
FAKE_MASK_VALUE = -1e7
PREV_CHOICE_MASK_VALUE = -1e9
FULL_ORDER_MASK_VALUE = -1e9


class DeliveryAction2(Action):
    def __init__(self, idxes: list[int]) -> None:
        self.idxes = idxes

    def to_indexes(self) -> list[int]:
        return self.idxes


class DeliveryState2(State):
    def __init__(self,
                 claim_embs: typing.Optional[np.ndarray],
                 couriers_embs: typing.Optional[np.ndarray],
                 orders_embs: typing.Optional[np.ndarray],
                 orders_full_masks: list[bool],
                 claims_to_couries_dists: np.ndarray
                 ) -> None:
        self.claim_embs = claim_embs
        self.claims_to_couries_dists = claims_to_couries_dists
        self.couriers_embs = couriers_embs
        self.orders_embs = orders_embs
        self.orders_full_masks = orders_full_masks

    def size(self) -> int:
        len_crr = len(self.couriers_embs) if self.couriers_embs is not None else 0
        len_ord = len(self.orders_embs) if self.orders_embs is not None else 0
        return len_crr + len_ord


class DeliveryRewarder2:
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


class DeliveryEnvironment2(BaseEnvironment):
    def __init__(self, simulator: Simulator, rewarder: DeliveryRewarder2, **kwargs) -> None:
        self.max_num_points_in_route = kwargs['max_num_points_in_route']
        self.num_gambles = kwargs['num_gambles_in_day']
        self.simulator = simulator
        self.device = kwargs['device']
        self.rewarder = rewarder

    def copy(self) -> 'DeliveryEnvironment2':
        return DeliveryEnvironment2(
            simulator=deepcopy(self.simulator),
            rewarder=self.rewarder,
            max_num_points_in_route=self.max_num_points_in_route,
            num_gambles_in_day=self.num_gambles,
            device=self.device,
        )

    def reset(self) -> DeliveryState2:
        self._iter = 0
        self._gamble: typing.Optional[Gamble] = None
        self._assignments: Assignment = Assignment([])
        self._assignment_statistics: dict[str, float] = {}
        self.simulator.reset()
        self._update_next_gamble()
        state = self._make_state_from_gamble_dict()
        return state

    def step(self, action: DeliveryAction2) -> tuple[DeliveryState2, float, bool, dict[str, float]]:
        if self.__getattribute__('_iter') is None:
            raise RuntimeError('Call reset before doing steps')
        # LOGGER.debug(f'fake assignment: {action.to_index() == len(self._gamble.orders) + len(self._gamble.couriers)}')
        self._update_assignments(action)
        self._update_next_gamble()
        reward = self.rewarder(self._assignment_statistics)
        info = self._assignment_statistics
        new_state = self._make_state_from_gamble_dict()
        done = self._iter == self.num_gambles
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
            if len(self._gamble.couriers) > 0
            else None,
            'clm': np.stack([clm.to_numpy() for clm in self._gamble.claims], axis=0),
            'ord': np.stack([ord.to_numpy(max_num_points_in_route=self.max_num_points_in_route)
                             for ord in self._gamble.orders], axis=0)
            if len(self._gamble.orders) > 0
            else None,
            'ord_masks': [ord.has_full_route(max_num_points_in_route=self.max_num_points_in_route)
                          for ord in self._gamble.orders],
            'dists': compulte_claims_to_couriers_distances(self._gamble),
        }
        self._assignments = Assignment([])
        self._iter += 1

    def _update_assignments(self, action: DeliveryAction2):
        for c_idx, co_idx in enumerate(action.to_indexes()):
            if co_idx < len(self._gamble.couriers):
                self._assignments.ids.append((
                    self._gamble.couriers[co_idx].id,
                    self._gamble.claims[c_idx].id
                ))
            elif co_idx - len(self._gamble.couriers) < len(self._gamble.orders):
                self._assignments.ids.append((
                    self._gamble.orders[co_idx - len(self._gamble.couriers)].courier.id,
                    self._gamble.claims[c_idx].id
                ))

    def _make_state_from_gamble_dict(self) -> DeliveryState2:
        return DeliveryState2(
            claim_embs=self.embs_dict['clm'],
            claims_to_couries_dists=self.embs_dict['dists'],
            couriers_embs=self.embs_dict['crr'],
            orders_embs=self.embs_dict['ord'],
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


class DeliveryActorCritic2(BaseActorCritic):
    def __init__(self,
                 gamble_encoder: GambleEncoder,
                 backbone: TransformerBackbone,
                 clm_emb_size: int,
                 crr_ord_emb_size: int,
                 temperature: float,
                 device,
                 mask_fake_crr: bool = False,
                 use_dist_feature: bool = False
                 ) -> None:
        super().__init__()
        self.gamble_encoder = gamble_encoder
        self.backbone = backbone
        self.clm_emb_size = clm_emb_size
        self.crr_ord_emb_size = crr_ord_emb_size
        self.temperature = temperature
        self.mask_fake_crr = mask_fake_crr
        self.use_dist_feature = use_dist_feature
        self.device = device

        if self.use_dist_feature:
            raise NotImplementedError

    def forward(self, state_list: list[DeliveryState2]) -> None:
        policy_tens, val_tens = self._make_padded_policy_value_tensors(state_list)
        assert (policy_tens.isnan().all(dim=-1)).sum() == 0, policy_tens
        self.log_probs = nn.functional.log_softmax(policy_tens / self.temperature, dim=-1)
        self.values = val_tens
        self._actions = None

    def _make_padded_policy_value_tensors(self, states: list[DeliveryState2]
                                          ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # bs = len(states)
        # max_clm_len, max_co_len = 0, 0
        # for state in states:
        #     max_clm_len = max(max_clm_len, len(state.claim_embs) if state.claim_embs is not None else 0)
        #     crr_len = len(state.couriers_embs) if state.couriers_embs is not None else 0
        #     ord_len = len(state.orders_embs) if state.orders_embs is not None else 0
        #     max_co_len = max(max_co_len, crr_len + ord_len + 1)
        # clm_embs_tensor = torch.zeros(size=(bs, max_clm_len, self.clm_emb_size), device=self.device, dtype=torch.float)
        # crr_ord_embs_tensor = torch.zeros(size=(bs, max_co_len, self.crr_ord_emb_size),
        #                                   device=self.device, dtype=torch.float)
        # full_masks_tensor = torch.zeros(size=(bs, max_co_len), device=self.device, dtype=torch.float)
        clm_embs_list, crr_ord_embs_list, attn_masks_list, pad_clm_masks_list, pad_co_masks_list = [], [], [], [], []
        for idx, state in enumerate(states):
            co_embs, clm_embs = self._make_three_tensors_from_state(state)
            # if self.use_dist_feature:
            #     co_dists = torch.tensor(state.claims_to_couries_dists,
            #                             dtype=torch.float, device=self.device).unsqueeze(-1)
            #     co_embs = torch.cat([co_embs, co_dists], dim=-1)
            attn_masks_list.append(self._make_full_order_mask(state))
            crr_ord_embs_list.append(co_embs)
            clm_embs_list.append(clm_embs)
            pad_clm_masks_list.append(torch.tensor([False] * len(clm_embs), device=self.device))
            pad_co_masks_list.append(torch.tensor([False] * len(co_embs), device=self.device))
            # crr_len = len(state.couriers_embs) if state.couriers_embs is not None else 0
            # ord_len = len(state.orders_embs) if state.orders_embs is not None else 0
            # clm_embs_tensor[idx, :clm_embs.size(0), :] = clm_embs
            # crr_ord_embs_tensor[idx, :co_embs.size(0), :] = co_embs
            # full_masks_tensor[idx, :full_ord_masks.size(0)] = FULL_ORDER_MASK_VALUE

        crr_ord_embs_tensor = pad_sequence(crr_ord_embs_list, batch_first=True)
        clm_embs_tensor = pad_sequence(clm_embs_list, batch_first=True)
        attn_masks_tensor = pad_sequence(attn_masks_list, batch_first=True, padding_value=True)
        bs, nhead = len(states), self.backbone.nhead
        attn_masks_tensor = attn_masks_tensor.view(bs, 1, -1).repeat(1, nhead, 1).view(bs * nhead, -1)
        attn_masks_tensor = (attn_masks_tensor.unsqueeze(-1) | attn_masks_tensor.unsqueeze(-2))
        attn_masks_tensor = torch.zeros(size=attn_masks_tensor.shape, device=self.device
                                        ).masked_fill_(attn_masks_tensor, value=FULL_ORDER_MASK_VALUE)
        pad_clm_masks_tensor = pad_sequence(pad_clm_masks_list, batch_first=True, padding_value=True)
        pad_co_masks_tensor = pad_sequence(pad_co_masks_list, batch_first=True, padding_value=True)
        assert attn_masks_tensor.dtype == torch.float, attn_masks_tensor.dtype

        policy_tens_result, value_tens_result = self.backbone(crr_ord_embs_tensor, clm_embs_tensor, attn_masks_tensor,
                                                              pad_clm_masks_tensor, pad_co_masks_tensor)
        # assert (~policy_tens_result.isnan().all(dim=-1)).sum() == 0, policy_tens_result
        policy_tens_result += pad_clm_masks_tensor.unsqueeze(-1) * PAD_MASK_VALUE
        policy_tens_result += pad_co_masks_tensor.unsqueeze(-2) * PAD_MASK_VALUE
        return policy_tens_result, value_tens_result

    def _make_full_order_mask(self, state: DeliveryState2) -> torch.BoolTensor:
        couriers_part = [False] * len(state.couriers_embs) if state.couriers_embs is not None else []
        orders_part = state.orders_full_masks
        mask = torch.tensor(couriers_part + orders_part + [False], device=self.device, dtype=torch.bool)
        return mask

    def _make_three_tensors_from_state(self, state: DeliveryState2
                                       ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Returns non-zero tensors of couriers-orders-fake_courier embeds and claims embeds
        '''
        embs_dict = {
            'clm': state.claim_embs,
            'crr': state.couriers_embs,
            'ord': state.orders_embs,
        }
        encoded_dict = self.gamble_encoder(embs_dict)
        fake_crr = torch.ones(size=(1, self.gamble_encoder.courier_encoder.item_embedding_dim), device=self.device)
        co_embs = torch.cat(
            ([encoded_dict['crr']] if encoded_dict['crr'] is not None else []) +
            ([encoded_dict['ord']] if encoded_dict['ord'] is not None else []) +
            [fake_crr], dim=0)
        clm_embs = encoded_dict['clm']
        return co_embs, clm_embs

    def get_actions_list(self, best_actions=False) -> list[DeliveryAction2]:
        if best_actions:
            self._actions = torch.argmax(self.log_probs, dim=-1)
        else:
            self._actions = torch.distributions.categorical.Categorical(logits=self.log_probs).sample()
        return [DeliveryAction2(a.tolist()) for a in self._actions]

    def get_log_probs_list(self) -> list[float]:
        bs, clm_len, co_len = self.log_probs.shape
        arange = torch.arange(bs * clm_len).to(self.device)
        actions = self._actions.view(bs * clm_len)
        probs = self.log_probs.view(bs * clm_len, co_len)[arange, actions].view(bs, clm_len)
        return probs.tolist()

    def get_values_list(self) -> list[float]:
        return self.values.tolist()

    def get_log_probs_tensor(self) -> torch.FloatTensor:
        return self.log_probs

    def get_values_tensor(self) -> torch.FloatTensor:
        return self.values
