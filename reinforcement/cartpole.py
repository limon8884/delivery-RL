import gym
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(10, 2)
        self.value_head = nn.Linear(10, 1)

    def forward(self, x):
        x = self.f(x)
        return self.action_head(x), self.value_head(x)

class ActorCartPole:
    def __init__(self, net) -> None:
        self.net = net

    def __call__(self, state):
        state = torch.tensor(state, dtype=torch.float)
        logits, _ = self.net(state)
        action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
        action = np.random.choice([0, 1], size=None, p=action_probs)
        return action

class EnvCartPole:
    def __init__(self, actor) -> None:
        self.env = gym.make("CartPole-v1")
        self.actor = actor
        self.state = self.env.reset()
        self.reward = 0
        self.action = None

    def Next(self):
        self.action = self.actor(self.state)
        self.state, self.reward, done, info = self.env.step(self.action)
        if done:
            self.reward -= 10
            self.state = self.env.reset()

    def GetState(self):
        return self.state
    
    def GetReward(self):
        return self.reward
    
    def GetAction(self):
        return self.action