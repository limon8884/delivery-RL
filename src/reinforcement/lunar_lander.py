import typing
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
from copy import deepcopy

from src.reinforcement.base import (
    Action,
    State,
    BaseEnvironment,
    BaseActorCritic,
    Runner,
    GAE,
    Buffer,
    PPO,
    Logger,
    TrajectorySampler,
    RewardNormalizer,
    InferenceMetricsRunner,
    BaseMaker,
    make_optimizer,
)


class GymAction(Action):
    def __init__(self, value: typing.Any) -> None:
        self.value = value
        # assert isinstance(self.value, int)

    def to_index(self) -> int:
        return self.value


class GymState(State):
    def __init__(self, value: typing.Any) -> None:
        self.value = value


class GymEnv(BaseEnvironment):
    def __init__(self, gym_name: str) -> None:
        self.env = gym.make(gym_name)
        self.action_dim = self.env.action_space.n
        assert len(self.env.observation_space.shape) == 1
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action: GymAction) -> tuple[GymState, float, bool, dict[str, float]]:
        state, reward, reset, _, _ = self.env.step(action.value)
        state = GymState(state)
        if reset:
            state = self.reset()
        return state, float(reward), reset, {}

    def reset(self, seed: typing.Optional[int] = None) -> GymState:
        if seed is None:
            return GymState(self.env.reset()[0])
        return GymState(self.env.reset(seed=seed)[0])

    def copy(self) -> 'GymEnv':
        return deepcopy(self)


class GymActorCritic(BaseActorCritic):
    def __init__(self, input_dim: int, action_dim: int, device) -> None:
        super().__init__()
        self.hidden_dim = 64
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.policy_model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, action_dim),
        ).to(device=device)

        self.value_model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        ).to(device=device)
        self.device = device

    def forward(self, state_list: list[State]) -> None:
        # inp = torch.stack([torch.FloatTensor(state.value).to(self.device) for state in state_list], dim=0)
        inp = torch.stack([torch.tensor(state.value, dtype=torch.float).to(self.device) for state in state_list], dim=0)
        policy = self.policy_model(inp)
        self.log_probs = nn.functional.log_softmax(policy, dim=-1)
        self.actions = torch.distributions.categorical.Categorical(logits=self.log_probs).sample()
        values = self.value_model(inp)
        self.values = values.squeeze(-1)

    def get_actions_list(self, best_actions=False) -> list[Action]:
        if best_actions:
            return [GymAction(a.item()) for a in torch.argmax(self.log_probs, dim=-1)]
        return [GymAction(a.item()) for a in self.actions]

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


class GymMaker(BaseMaker):
    def __init__(self, **kwargs) -> None:
        batch_size = kwargs['batch_size']
        device = kwargs['device']

        self._train_logger = Logger(use_wandb=kwargs['use_wandb'])
        self._env = GymEnv(gym_name=kwargs['env_name'])
        self._ac = GymActorCritic(input_dim=self._env.state_dim, action_dim=self._env.action_dim, device=device)
        opt = make_optimizer(self._ac.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=kwargs['scheduler_max_lr'],
                                                        total_steps=kwargs['total_iters'],
                                                        pct_start=kwargs['scheduler_pct_start'])
        self._ppo = PPO(actor_critic=self._ac, opt=opt, scheduler=scheduler,
                        logger=self._train_logger, **kwargs)
        runner = Runner(environment=self._env, actor_critic=self._ac,
                        n_envs=kwargs['n_envs'], trajectory_lenght=kwargs['trajectory_lenght'])
        # inference_logger = InferenceMetricsRunner(runner=runner, logger=self._train_logger)
        gae = GAE(gamma=kwargs['gae_gamma'], lambda_=kwargs['gae_lambda'])
        normalizer = RewardNormalizer()
        buffer = Buffer(gae=gae, reward_normalizer=normalizer, device=device)
        self._sampler = TrajectorySampler(runner, buffer, num_epochs_per_traj=kwargs['num_epochs_per_traj'],
                                          batch_size=batch_size)

    @property
    def ppo(self) -> PPO:
        return self._ppo

    @property
    def sampler(self) -> TrajectorySampler:
        return self._sampler

    @property
    def actor_critic(self) -> GymActorCritic:
        return self._ac

    @property
    def environment(self) -> GymEnv:
        return self._env

    @property
    def logger(self) -> Logger:
        return self._train_logger
