import typing
import torch
import json
import wandb
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from copy import deepcopy

from src.objects import (
    Claim,
    Courier,
    Order,
    Gamble,
    Assignment,
)
from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.router_makers import AppendRouteMaker
from src.database.database import Database, Metric, Logger as DB_Logger
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
    Logger as TrainLogger,
    TrajectorySampler,
    RewardNormalizer,
    InferenceMetricsRunner,
    make_optimizer,
)


class DeliveryAction(Action):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def to_index(self) -> int:
        return self.idx


class DeliveryState(State):
    def __init__(self, claim_emb: np.ndarray, couriers_embs: typing.Optional[np.ndarray],
                 orders_embs: typing.Optional[np.ndarray]) -> None:
        self.claim_emb = claim_emb
        self.couriers_embs = couriers_embs
        self.orders_embs = orders_embs


class DeliveryEnvironment(BaseEnvironment):
    def __init__(self, simulator: Simulator, max_num_points_in_route: int, num_gambles: int, device) -> None:
        self.max_num_points_in_route = max_num_points_in_route
        self.num_gambles = num_gambles
        self.simulator = simulator
        self.device = device
        self.rewarder: typing.Callable[[dict[str, float]], float] = \
            lambda d: d['completed_claims'] + 0.1 * (d['assigned_not_batched_claims'] + d['assigned_batched_claims']) \
            - d['cancelled_claims']
        self.reset()

    def copy(self) -> 'DeliveryEnvironment':
        return DeliveryEnvironment(
            simulator=deepcopy(self.simulator),
            max_num_points_in_route=self.max_num_points_in_route,
            num_gambles=self.num_gambles,
            device=self.device,
        )

    def reset(self) -> DeliveryState:
        self._iter = 0
        self._gamble: typing.Optional[Gamble] = None
        self._claim_idx: int = 0
        self._assignments: Assignment = Assignment([])
        self._base_gamble_reward: float = 0.0
        self.simulator.reset()
        self._update_next_gamble()
        state = self._make_state_from_gamble_dict()
        return state

    def step(self, action: DeliveryAction) -> tuple[DeliveryState, float, bool, dict[str, float]]:
        self._update_assignments(action)
        reward = 0
        done = False
        if self._claim_idx == len(self.embs_dict['clm']):
            self._update_next_gamble()
            reward = self.rewarder(self.simulator.assignment_statistics)
        new_state = self._make_state_from_gamble_dict()
        if self._iter == self.num_gambles:
            done = True
        return new_state, reward, done, self.simulator.assignment_statistics

    def _update_next_gamble(self):
        self.simulator.next(self._assignments)
        self._gamble = self.simulator.get_state()
        while len(self._gamble.claims) == 0:
            self.simulator.next(Assignment([]))
            self._gamble = self.simulator.get_state()
        self.embs_dict = {
            'crr': np.stack([crr.to_numpy() for crr in self._gamble.couriers], axis=0)
            if len(self._gamble.couriers)
            else None,
            'clm': np.stack([clm.to_numpy() for clm in self._gamble.claims], axis=0),
            'ord': np.stack([ord.to_numpy(max_num_points_in_route=self.max_num_points_in_route)
                             for ord in self._gamble.orders], axis=0)
            if len(self._gamble.orders) > 0
            else None,
        }
        self._assignments = Assignment([])
        self._claim_idx = 0
        self._iter += 1

    def _update_assignments(self, action: DeliveryAction):
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

    def _make_state_from_gamble_dict(self) -> DeliveryState:
        claim_emb = self.embs_dict['clm'][self._claim_idx]
        return DeliveryState(
            claim_emb=claim_emb,
            couriers_embs=self.embs_dict['crr'],
            orders_embs=self.embs_dict['ord']
        )


class DeliveryActorCritic(BaseActorCritic):
    def __init__(self, gamble_encoder: GambleEncoder, clm_emb_size: int, temperature: float, device) -> None:
        super().__init__()
        self.gamble_encoder = gamble_encoder
        self.clm_emb_size = clm_emb_size
        self.temperature = temperature
        self.device = device

    def forward(self, state_list: list[DeliveryState]) -> None:
        pol_tens, val_tens, clm_tens = self._make_masked_policy_value_claim_tensors(state_list)
        policy = (clm_tens.unsqueeze(1) @ pol_tens.transpose(-1, -2)).squeeze(1)
        self.log_probs = nn.functional.log_softmax(policy / self.temperature, dim=-1)
        self.actions = torch.distributions.categorical.Categorical(logits=self.log_probs).sample()
        self.values = (clm_tens @ torch.mean(val_tens, dim=1).T).diag()

    def _make_masked_policy_value_claim_tensors(self, states: list[DeliveryState]
                                                ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        policy_tens_list, value_tens_list, claim_tens_list = [], [], []
        for state in states:
            policy_tens, value_tens, claim_emb = self._make_masked_policy_value_claim_tensors_from_state(state)
            policy_tens_list.append(policy_tens)
            value_tens_list.append(value_tens)
            claim_tens_list.append(claim_emb)
        policy_tens_result = pad_sequence(policy_tens_list, batch_first=True, padding_value=-1e11)
        value_tens_result = pad_sequence(value_tens_list, batch_first=True, padding_value=0.0)
        claim_tens_result = torch.stack(claim_tens_list, dim=0)
        return policy_tens_result, value_tens_result, claim_tens_result

    def _make_masked_policy_value_claim_tensors_from_state(self, state: DeliveryState
                                                           ) -> tuple[torch.FloatTensor,
                                                                      torch.FloatTensor, torch.FloatTensor]:
        embs_dict = {
            'clm': state.claim_emb.reshape(1, -1),
            'crr': state.couriers_embs,
            'ord': state.orders_embs,
        }
        encoded_dict = self.gamble_encoder(embs_dict)
        fake_crr = torch.zeros(size=(1, self.gamble_encoder.courier_encoder.item_embedding_dim), device=self.device)
        co_embs = torch.cat(
            ([encoded_dict['crr']] if encoded_dict['crr'] is not None else []) +
            ([encoded_dict['ord']] if encoded_dict['ord'] is not None else []) +
            [fake_crr], dim=-2)
        return co_embs[:, :self.clm_emb_size], co_embs[:, self.clm_emb_size:], encoded_dict['clm'][0]

    def _make_clm_tens(self, clm_emb_list: list[np.ndarray]) -> torch.FloatTensor:
        # clm_embs_tens_list = [torch.FloatTensor(clm_emb, device=self.device) for clm_emb in clm_emb_list]
        clm_embs_tens_list = [torch.tensor(clm_emb, dtype=torch.float, device=self.device) for clm_emb in clm_emb_list]
        return torch.stack(clm_embs_tens_list, dim=0)

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


def run_ppo(**kwargs):
    if kwargs['use_wandb']:
        wandb.login()
        wandb.init(
            project="delivery-RL-v2",
            name=f"run_{kwargs['run_id']}",
            config=kwargs
        )

    batch_size = kwargs['batch_size']
    max_num_points_in_route = kwargs['max_num_points_in_route']
    device = kwargs['device']

    simulator_config_path = Path(kwargs['simulator_cfg_path'])
    network_config_path = Path(kwargs['network_cfg_path'])
    db_path = Path(kwargs['database_logger_path'])
    db = Database(db_path)
    db.clear()

    with open(network_config_path) as f:
        encoder_cfg = json.load(f)['encoder']
    gamble_encoder = GambleEncoder(max_num_points_in_route=max_num_points_in_route, **encoder_cfg, device=device)
    db_logger = DB_Logger(run_id=kwargs['run_id'])
    reader = DataReader.from_config(config_path=simulator_config_path,
                                    sampler_mode=kwargs['sampler_mode'], logger=db_logger)
    route_maker = AppendRouteMaker(max_points_lenght=max_num_points_in_route, cutoff_radius=0.0)
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=simulator_config_path, logger=db_logger)

    train_logger = TrainLogger(use_wandb=kwargs['use_wandb'])
    env = DeliveryEnvironment(simulator=sim, max_num_points_in_route=max_num_points_in_route,
                              num_gambles=kwargs['num_gambles_in_day'], device=device)
    ac = DeliveryActorCritic(gamble_encoder=gamble_encoder, clm_emb_size=encoder_cfg['claim_embedding_dim'],
                             temperature=kwargs['exploration_temperature'], device=device)
    optimizer = make_optimizer(ac.parameters(), **kwargs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=kwargs['scheduler_max_lr'],
                                                    total_steps=kwargs['total_iters'],
                                                    pct_start=kwargs['scheduler_pct_start'])
    ppo = PPO(actor_critic=ac, optimizer=optimizer, device=device, scheduler=scheduler, logger=train_logger,
              cliprange=kwargs['ppo_cliprange'], value_loss_coef=kwargs['ppo_value_loss_coef'],
              max_grad_norm=kwargs['max_grad_norm'])
    runner = Runner(environment=env, actor_critic=ac,
                    n_envs=kwargs['n_envs'], trajectory_lenght=kwargs['trajectory_lenght'])
    eval_runner = Runner(environment=env, actor_critic=ac,
                         n_envs=kwargs['eval_n_envs'], trajectory_lenght=kwargs['eval_trajectory_lenght'])
    gae = GAE(gamma=kwargs['gae_gamma'], lambda_=kwargs['gae_lambda'])
    normalizer = RewardNormalizer()
    buffer = Buffer(gae, reward_normalizer=normalizer, device=device)
    sampler = TrajectorySampler(runner, buffer, num_epochs_per_traj=kwargs['num_epochs_per_traj'],
                                batch_size=batch_size)
    inference_logger = InferenceMetricsRunner(runner=eval_runner, logger=train_logger)

    for iteration in tqdm(range(kwargs['total_iters'])):
        ac.train()
        sample = sampler.sample()
        ppo.step(sample)
        if iteration % kwargs['eval_epochs_frequency'] == 0:
            ac.eval()
            inference_logger()
            if not kwargs['use_wandb']:
                train_logger.plot(window_size=10)
            torch.save(ac.state_dict(), kwargs['checkpoint_path'])
