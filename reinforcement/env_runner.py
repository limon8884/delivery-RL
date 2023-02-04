""" RL env runner """
from collections import defaultdict
import torch
import numpy as np


class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, n_steps, n_envs = 8, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.transforms = transforms or [ComputeValueTargets(policy=self.policy), MergeTimeBatch()]
        # self.step_var = step_var if step_var is not None else 0
        self.state = {"latest_observation": self.env.reset()}

    def reset(self):
        """ Resets env and runner states. """
        self.state["latest_observation"] = self.env.reset()
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.n_steps

        for i in range(self.n_steps):
            observations.append(self.state["latest_observation"])
            act = self.policy.act(self.state["latest_observation"])
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, done, _ = self.env.step(trajectory["actions"][-1])
            self.state["latest_observation"] = obs
            rewards.append(rew)
            resets.append(done)
            # self.step_var += self.nenvs or 1

        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets)
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)
        return trajectory
    

class ComputeValueTargets:
    def __init__(self, policy, gamma=0.99):
        self.policy = policy
        self.gamma = gamma

    def __call__(self, trajectory):
        # This method should modify trajectory inplace by adding
        # an item with key 'value_targets' to it.
        # <Compute value targets for a given partial trajectory>
        T = len(trajectory['observations'])
        # v_last = self.policy.act(torch.tensor(trajectory['state']['latest_observation']).unsqueeze(-1))
        latest_obs = torch.tensor(trajectory['state']['latest_observation'], dtype=torch.float).transpose(1, 3)
        a_last, v_last = self.policy.model(latest_obs)
        v_pred_last = v_last.detach()

        rewards = trajectory['rewards']
        v_list = [v_pred_last]
        for t in range(T)[::-1]:
            v_list.append(torch.tensor( v_list[-1] * self.gamma * np.logical_not(trajectory['resets'][t]).astype(float) + rewards[t] ).detach())

        trajectory['value_targets'] = v_list[1:][::-1]

class MergeTimeBatch:
    """ Merges first two axes typically representing time and env batch. """
    def __call__(self, trajectory):
        pass
        # Modify trajectory inplace.
        # <TODO: implement>

        # FIX reimplement
        # trajectory['logits'] = torch.hstack(trajectory['logits'])
        # trajectory['log_probs'] = torch.hstack(trajectory['log_probs'])
        # trajectory['values'] = torch.hstack(trajectory['values'])
        # trajectory['value_targets'] = torch.hstack(trajectory['value_targets'])