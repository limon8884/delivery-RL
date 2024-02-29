import typing
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
from copy import deepcopy

from src_new.reinforcement.base import (
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


# def avg_traj_reward_length(ac: GymActorCritic, env: GymEnv, n_envs=8):
#     total_rew = 0.0
#     total_length = 0
#     for _ in range(n_envs):
#         env_this = env.copy()
#         state = env_this.reset()
#         for t in range(2048):
#             ac([state])
#             act = ac.get_actions_list(best_actions=True)[0]
#             state, rew, done = env_this.step(act)
#             total_rew += rew
#             total_length += 1
#             if done:
#                 break
#     return total_rew / n_envs, total_length / n_envs


def run_ppo():
    n_envs = 1
    trajectory_lenght = 2048
    batch_size = 64
    num_epochs_per_traj = 10
    total_iters = 250000
    device = None
    env_name = 'LunarLander-v2'

    logger = Logger()
    env = GymEnv(gym_name=env_name)
    ac = GymActorCritic(input_dim=env.state_dim, action_dim=env.action_dim, device=device)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4, eps=1e-5)
    ppo = PPO(actor_critic=ac, optimizer=opt, device=device, logger=logger)
    runner = Runner(environment=env, actor_critic=ac,
                    n_envs=n_envs, trajectory_lenght=trajectory_lenght)
    inference_logger = InferenceMetricsRunner(runner=runner, logger=logger)
    gae = GAE()
    normalizer = RewardNormalizer()
    buffer = Buffer(gae=gae, reward_normalizer=normalizer, device=device)
    sampler = TrajectorySampler(runner, buffer, num_epochs_per_traj=num_epochs_per_traj, batch_size=batch_size)

    for iteration in tqdm(range(total_iters)):
        ac.train()
        sample = sampler.sample()
        ppo.step(sample)
        if iteration % 1000 == 0:
            ac.eval()
            inference_logger()
            logger.plot()


if __name__ == '__main__':
    run_ppo()
