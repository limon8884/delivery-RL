import typing
import torch
from torch import nn
import numpy as np
from copy import deepcopy

import gymnasium as gym

from src.reinforcement.base import (
    Action,
    State,
    BaseEnvironment,
    BaseActorCritic,
    Runner,
    Trajectory,
    GAE,
    Buffer,
    PPO,
)


class TestAction(Action):
    def __init__(self, value: typing.Any) -> None:
        self.value = value
        assert isinstance(self.value, int)

    def to_index(self) -> int:
        return self.value


class TestState(State):
    def __init__(self, value: typing.Any) -> None:
        self.value = value


class TestEnv(BaseEnvironment):
    def __init__(self) -> None:
        self.env = gym.make("LunarLander-v2")

    def step(self, action: TestAction) -> tuple[TestState, float, bool, dict[str, float]]:
        state, reward, reset, _, _ = self.env.step(action.value)
        return TestState(state), float(reward), reset, {}

    def reset(self) -> State:
        return TestState(self.env.reset()[0])

    def copy(self) -> BaseEnvironment:
        return deepcopy(self)


def test_interaction():
    env = TestEnv()
    _ = env.reset()
    action = TestAction(0)
    new_state, rew, done, _ = env.step(action)
    assert len(new_state.value) == 8


def test_trajectory():
    env = TestEnv()
    tr = Trajectory(env.reset())
    tr.append(
        TestState(np.array([0.0] * 8)),
        TestAction(3),
        0,
        False,
        -3.0,
        2.0,
    )
    assert tr.lenght == 1


class TestActorCritic(BaseActorCritic):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )

    def forward(self, state_list: list[State]) -> None:
        # inp = torch.stack([torch.FloatTensor(state.value) for state in state_list], dim=0)
        inp = torch.stack([torch.tensor(state.value, dtype=torch.float) for state in state_list], dim=0)
        out = self.net(inp)
        self.log_probs = nn.functional.softmax(out[:, :4], dim=-1)
        self.values = out[:, 4]

    def get_actions_list(self, best_actions=False) -> list[Action]:
        a = torch.argmax(self.log_probs, dim=-1)
        return [TestAction(e.item()) for e in a]

    def get_log_probs_list(self) -> list[float]:
        a = torch.max(self.log_probs, dim=-1)[0]
        return [e.item() for e in a]

    def get_values_list(self) -> list[float]:
        return [e.item() for e in self.values]

    def get_log_probs_tensor(self) -> torch.FloatTensor:
        return self.log_probs

    def get_values_tensor(self) -> torch.FloatTensor:
        return self.values


def test_actor_critic():
    ac = TestActorCritic()
    states = [
        TestState(np.zeros(8)),
        TestState(np.ones(8)),
    ]
    ac(states)
    states = [
        TestEnv().reset(),
        TestEnv().reset(),
    ]
    ac(states)
    lp_list = ac.get_log_probs_list()
    assert len(lp_list) == 2 and isinstance(lp_list[0], float)
    val_list = ac.get_values_list()
    assert len(val_list) == 2 and isinstance(val_list[0], float)
    act_tens = ac.get_log_probs_tensor()
    assert act_tens.shape == (2, 4)
    val_tens = ac.get_values_tensor()
    assert val_tens.shape == (2,)
    act_list = ac.get_actions_list()
    assert len(act_list) == 2 and isinstance(act_list[0], TestAction)


def test_runner():
    runner = Runner(environment=TestEnv(), actor_critic=TestActorCritic(),
                    n_envs=4, trajectory_lenght=20)
    runner.reset()
    trajs = runner.run()
    assert len(trajs) == 4
    assert isinstance(trajs[0], Trajectory)
    for traj in trajs:
        assert traj.lenght == 20


def test_gae():
    runner = Runner(environment=TestEnv(), actor_critic=TestActorCritic(),
                    n_envs=1, trajectory_lenght=100)
    runner.reset()
    traj = runner.run()[0]
    gae = GAE()
    gaes = gae(traj)
    assert len(gaes) == 100


def test_buffer():
    runner = Runner(environment=TestEnv(), actor_critic=TestActorCritic(),
                    n_envs=2, trajectory_lenght=100)
    runner.reset()
    trajs = runner.run()
    gae = GAE()
    buffer = Buffer(gae, device=None)
    buffer.update(trajs)
    sample = buffer.sample(4)
    assert sample['advantages'].shape == (4,)
    assert sample['log_probs_chosen'].shape == (4,)
    assert sample['actions_chosen'].shape == (4,)
    assert sample['values'].shape == (4,)
    assert len(sample['states']) == 4 and isinstance(sample['states'][0], TestState)


def test_ppo():
    ac = TestActorCritic()
    ac.train()
    opt = torch.optim.SGD(ac.parameters(), lr=3e-4)
    ppo = PPO(actor_critic=ac, opt=opt, device=None, ppo_cliprange=0.2, ppo_value_loss_coef=0.25,
              ppo_entropy_loss_coef=0.01, max_grad_norm=1.0, debug_info_path='')
    runner = Runner(environment=TestEnv(), actor_critic=ac,
                    n_envs=4, trajectory_lenght=10)
    runner.reset()
    trajs = runner.run()
    gae = GAE()
    buffer = Buffer(gae, device=None)
    buffer.update(trajs)
    sample = buffer.sample(8)
    ppo.step(sample)
