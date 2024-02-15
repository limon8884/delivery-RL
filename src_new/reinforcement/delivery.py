import typing
import torch
import numpy as np
import json
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from copy import deepcopy

from src_new.objects import (
    Claim,
    Courier,
    Order,
    Gamble,
    Assignment,
)
from src_new.simulator.simulator import Simulator
from src_new.simulator.data_reader import DataReader
from src_new.router_makers import AppendRouteMaker
from src_new.database.database import Database, Metric, Logger as DB_Logger
from src_new.networks.encoders import GambleEncoder


from src_new.reinforcement.base import (
    Action,
    State,
    BaseEnvironment,
    BaseActorCritic,
    Runner,
    GAE,
    Buffer,
    PPO,
    Logger as TrainLogger,
    TrajectorySampler,
    RewardNormalizer,
    InferenceMetricsRunner,
)


class DeliveryAction(Action):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def to_index(self) -> int:
        return self.idx


class DeliveryState(State):
    def __init__(self, claim_emb: torch.FloatTensor, co_embs: list[torch.FloatTensor]) -> None:
        self.claim_emb = claim_emb
        self.co_embs = co_embs


class DeliveryEnvironment(BaseEnvironment):
    def __init__(self, simulator: Simulator, gamble_encoder: GambleEncoder, num_gambles: int, device) -> None:
        self.num_gambles = num_gambles
        self.simulator = simulator
        self.gamble_encoder = gamble_encoder
        self.device = device
        self.rewarder: typing.Callable[[dict[str, float]], float] = \
            lambda d: d['completed_claims'] + 0.1 * d['assigned_claims']
        self.reset()

    def copy(self) -> 'DeliveryEnvironment':
        return DeliveryEnvironment(
            simulator=deepcopy(self.simulator),
            gamble_encoder=self.gamble_encoder,
            num_gambles=self.num_gambles,
            device=self.device,
        )

    def step(self, action: Action) -> tuple[DeliveryState, float, bool]:
        if self._claim_embs is None or self._claim_idx == len(self._claim_embs):
            self._update_next_gamble()
            self._base_gamble_reward = self.rewarder(self.simulator.assignment_statistics)
        self._make_assignment(action)
        reward = self._base_gamble_reward
        done = self._iter == self.num_gambles and self._claim_idx == len(self._claim_embs)
        return DeliveryState(claim_emb=self._claim_embs[self._claim_idx, :], co_embs=self._co_embs), reward, done

    def reset(self) -> DeliveryState:
        self._iter = 0
        self._gamble: typing.Optional[Gamble] = None
        self._claim_embs: typing.Optional[torch.FloatTensor] = None
        self._co_embs: typing.Optional[torch.FloatTensor] = None
        self._claim_idx: int = 0
        self._assignments: Assignment = Assignment([])
        self._base_gamble_reward: float = 0.0
        self._num_couriers: int = 0
        self.simulator.reset()
        self._update_next_gamble()
        return DeliveryState(claim_emb=self._claim_embs[0, :], co_embs=self._co_embs)

    def _update_next_gamble(self):
        self.simulator.next(self._assignments)
        self._gamble = self.simulator.get_state()
        while len(self._gamble.claims) == 0:
            self.simulator.next(Assignment([]))
            self._gamble = self.simulator.get_state()

        gamble_dict = {
            'crr': np.stack([crr.to_numpy() for crr in self._gamble.couriers], axis=0)
            if len(self._gamble.couriers)
            else None,
            'clm': np.stack([clm.to_numpy() for clm in self._gamble.claims], axis=0),
            'ord': np.stack([ord.to_numpy(max_num_points_in_route=10) for ord in self._gamble.orders], axis=0)
            if len(self._gamble.orders) > 0
            else None,
        }

        emb_dict = self.gamble_encoder(gamble_dict)
        self._claim_embs = emb_dict['clm']
        fake_crr = torch.zeros(size=(1, self.gamble_encoder.courier_encoder.item_embedding_dim),
                               device=self.device, dtype=torch.float)
        self._co_embs = torch.cat(
            ([emb_dict['crr']] if emb_dict['crr'] is not None else []) +
            ([emb_dict['ord']] if emb_dict['ord'] is not None else []) +
            [fake_crr], dim=-2)
        self._num_couriers = len(emb_dict['crr']) if emb_dict['crr'] is not None else 0
        self._assignments = Assignment([])
        self._claim_idx = 0
        self._iter += 1

    def _make_assignment(self, action: DeliveryAction):
        if action.idx < len(self._gamble.couriers):
            self._assignments.ids.append((
                self._gamble.couriers[action.idx].id,
                self._gamble.claims[self._claim_idx].id
            ))
        elif action.idx - len(self._gamble.couriers) < len(self._gamble.orders):
            self._assignments.ids.append((
                self._gamble.orders[action.idx - len(self._gamble.couriers)].courier.id,
                self._gamble.claims[self._claim_idx].id
            ))
        self._claim_idx += 1


class DeliveryActorCritic(BaseActorCritic):
    def __init__(self, clm_emb: int, device) -> None:
        super().__init__()
        self.clm_emb = clm_emb
        self.device = device

    def forward(self, state_list: list[DeliveryState]) -> None:
        pol_tens, val_tens = self._make_masked_policy_value_tensors([state.co_embs for state in state_list])
        clm_tens = self._make_clm_tens([state.claim_emb for state in state_list])

        policy = (clm_tens.unsqueeze(1) @ pol_tens.transpose(-1, -2)).squeeze(1)
        self.log_probs = nn.functional.log_softmax(policy, dim=-1)
        self.actions = torch.distributions.categorical.Categorical(logits=self.log_probs).sample()
        self.values = (clm_tens @ torch.mean(val_tens, dim=1).T).diag()

    def _make_masked_policy_value_tensors(self, co_emb_list: list[list[torch.FloatTensor]]
                                          ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        policy_tens_list, value_tens_list = [], []
        for co_embs in co_emb_list:
            policy_tens_list.append(
                torch.stack([co_emb[:self.clm_emb] for co_emb in co_embs], dim=0))
            value_tens_list.append(
                torch.stack([co_emb[self.clm_emb:] for co_emb in co_embs], dim=0))
        policy_tens = pad_sequence(policy_tens_list, batch_first=True, padding_value=-torch.inf)
        value_tens = pad_sequence(value_tens_list, batch_first=True, padding_value=0.0)
        return policy_tens, value_tens

    def _make_clm_tens(self, clm_emb_list: list[torch.FloatTensor]) -> torch.FloatTensor:
        return torch.stack(clm_emb_list, dim=0)

    def get_actions_list(self, best_actions=False) -> list[Action]:
        if best_actions:
            return [DeliveryAction(a.item()) for a in torch.argmax(self.log_probs, dim=-1)]
        return [DeliveryAction(a.item()) for a in self.actions]

    def get_log_probs_list(self) -> list[float]:
        return [
            a.item()
            for a in torch.gather(self.log_probs, dim=-1, index=self.actions.unsqueeze(-1)).to(self.device).squeeze(-1)
        ]

    def get_values_list(self) -> list[float]:
        return [e.item() for e in self.values]

    def get_log_probs_tensor(self) -> torch.FloatTensor:
        return self.log_probs

    def get_values_tensor(self) -> torch.FloatTensor:
        return self.values


def run_ppo():
    n_envs = 2
    trajectory_lenght = 128
    batch_size = 64
    num_epochs_per_traj = 10
    total_iters = 250000
    max_num_points_in_route = 8
    num_gambles_in_day = 2880
    device = None

    simulator_config_path = Path('configs_new/simulator.json')
    network_config_path = Path('configs_new/network.json')
    db_path = Path('history.db')
    db = Database(db_path)
    db.clear()

    with open(network_config_path) as f:
        encoder_cfg = json.load(f)['simple']
    gamble_encoder = GambleEncoder(max_num_points_in_route=max_num_points_in_route, **encoder_cfg, device=device)
    db_logger = DB_Logger(run_id=0)
    reader = DataReader.from_config(config_path=simulator_config_path, sampler_mode='dummy_sampler', logger=db_logger)
    route_maker = AppendRouteMaker(max_points_lenght=max_num_points_in_route, cutoff_radius=0.0)
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=simulator_config_path, logger=db_logger)

    train_logger = TrainLogger()
    env = DeliveryEnvironment(simulator=sim,
                              gamble_encoder=gamble_encoder, num_gambles=num_gambles_in_day, device=device)
    ac = DeliveryActorCritic(clm_emb=encoder_cfg['claim_embedding_dim'], device=device)
    opt = torch.optim.Adam(chain(ac.parameters(), gamble_encoder.parameters()), lr=3e-4, eps=1e-5)
    ppo = PPO(actor_critic=ac, optimizer=opt, device=device, logger=train_logger)
    runner = Runner(environment=env, actor_critic=ac,
                    n_envs=n_envs, trajectory_lenght=trajectory_lenght)
    gae = GAE()
    normalizer = RewardNormalizer()
    buffer = Buffer(gae, reward_normalizer=normalizer, device=device)
    sampler = TrajectorySampler(runner, buffer, num_epochs_per_traj=num_epochs_per_traj, batch_size=batch_size)
    inference_logger = InferenceMetricsRunner(runner=runner, logger=train_logger)

    for iteration in tqdm(range(total_iters)):
        ac.train()
        sample = sampler.sample()
        ppo.step(sample)
        if iteration % 1000 == 0:
            ac.eval()
            inference_logger()
            train_logger.plot()


if __name__ == '__main__':
    run_ppo()
