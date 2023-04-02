import gym
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, mode) -> None:
        super().__init__()
        assert mode in ("LunarLander-v2", "CartPole-v1", "MountainCar-v0")
        if mode == "CartPole-v1":
            self.num_inp = 4
            self.num_out = 2
            self.hidden_dim = 8
        elif mode == "LunarLander-v2":
            self.num_inp = 8
            self.num_out = 4
            self.hidden_dim = 30
        elif mode == 'MountainCar-v0':
            self.num_inp = 2
            self.num_out = 2
            self.hidden_dim = 10
        self.f = nn.Sequential(
            nn.Linear(self.num_inp, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(self.hidden_dim, self.num_out)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.f(x)
        return self.action_head(x), self.value_head(x)

class ActorCartPole:
    def __init__(self, net: MLP) -> None:
        self.net = net

    def __call__(self, state):
        state = torch.tensor(state, dtype=torch.float)
        logits, _ = self.net(state)
        action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
        action = np.random.choice(range(self.net.num_out), size=None, p=action_probs)
        return action

class EnvCartPole:
    def __init__(self, actor, mode) -> None:
        assert mode in ("LunarLander-v2", "CartPole-v1", "MountainCar-v0")
        self.mode = mode
        self.env = gym.make(mode)
        self.actor = actor
        self.state = self.env.reset()
        self.reward = 0
        self.action = None

    def Next(self):
        self.action = self.actor(self.state)
        self.state, self.reward, done, info = self.env.step(self.action)
        if done:
            if self.mode == "CartPole-v1":
                self.reward -= 10
            if self.state[0] >= 0.5:
                self.reward = 100
            self.state = self.env.reset()

    def GetState(self):
        return self.state
    
    def GetReward(self):
        return self.reward
    
    def GetAction(self):
        return self.action