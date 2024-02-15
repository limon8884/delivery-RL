import numpy as np

import gymnasium as gym
class Summaries(gym.Wrapper):
    """ Wrapper to write summaries. """
    def __init__(self, env):
        super().__init__(env)
        self.episode_counter = 0
        self.current_step_var = 0

        self.episode_rewards = []
        self.episode_lens = []

        self.current_reward = 0
        self.current_len = 0

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        self.current_reward += rew
        self.current_len += 1
        self.current_step_var += 1

        if terminated or truncated:
            self.episode_rewards.append((self.current_step_var, self.current_reward))
            self.episode_lens.append((self.current_step_var, self.current_len))

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_counter += 1

        self.current_reward = 0
        self.current_len = 0

        return self.env.reset(**kwargs)
    

""" MuJoCo env wrappers. """
# Adapted from https://github.com/openai/baselines
import gymnasium as gym
import numpy as np


class RunningMeanVar:
    """ Computes running mean and variance.

    Args:
      eps (float): a small constant used to initialize mean to zero and
        variance to 1.
      shape tuple(int): shape of the statistics.
    """
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps

    def update(self, batch):
        """ Updates the running statistics given a batch of samples. """
        if not batch.shape[1:] == self.mean.shape:
            raise ValueError(f"batch has invalid shape: {batch.shape}, "
                             f"expected shape {(None,) + self.mean.shape}")
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """ Updates the running statistics given their new values on new data. """
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count,
                                       batch_mean, batch_var, batch_count):
    """ Updates running mean statistics given a new batch. """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    new_var = (
        var * (count / tot_count)
        + batch_var * (batch_count / tot_count)
        + np.square(delta) * (count * batch_count / tot_count ** 2))
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalize(gym.Wrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """
    # pylint: disable=too-many-arguments

    def __init__(self, env, obs=True, ret=True,
                 clipobs=10., cliprew=10., gamma=0.99, eps=1e-8):
        super().__init__(env)
        self.obs_rmv = (RunningMeanVar(shape=self.observation_space.shape)
                        if obs else None)
        self.ret_rmv = RunningMeanVar(shape=()) if ret else None
        self.clipob = clipobs
        self.cliprew = cliprew
        self.ret = np.zeros(getattr(self.env.unwrapped, "nenvs", 1))
        self.gamma = gamma
        self.eps = eps

    def observation(self, obs):
        """ Preprocesses a given observation. """
        if not self.obs_rmv:
            return obs
        rmv_batch = (np.expand_dims(obs, 0)
                     if not hasattr(self.env.unwrapped, "nenvs")
                     else obs)
        self.obs_rmv.update(rmv_batch)
        obs = (obs - self.obs_rmv.mean) / np.sqrt(self.obs_rmv.var + self.eps)
        obs = np.clip(obs, -self.clipob, self.clipob)
        return obs

    def step(self, action):
        obs, rews, terminated, truncated, info = self.env.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self.observation(obs)
        if self.ret_rmv:
            self.ret_rmv.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rmv.var + self.eps),
                           -self.cliprew, self.cliprew)
        self.ret[terminated] = 0.
        return obs, rews, terminated, truncated, info

    def reset(self, **kwargs):
        self.ret = np.zeros(getattr(self.env.unwrapped, "nenvs", 1))
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    

from src_new.reinforcement.lunar_lander import GymActorCritic, GymState
import torch

class Policy:
    def __init__(self, model: GymActorCritic) -> None:
        self.model = model

    def act(self, inputs, training=False):
        states = [GymState(inputs)]
        self.model(states)
        if not training:
            return {'actions': np.array(self.model.get_actions_list()[0].to_index()),
                    'log_probs': np.array(self.model.get_log_probs_list()[0]),
                    'values': np.array(self.model.get_values_list()[0])}
        else:
            return {'distribution': self.model.get_log_probs_tensor()[0], 'values': self.model.get_values_tensor()[0]}
    
    def reset(self):
        pass
        
class AsArray:
    """
    Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory):
        # Modify trajectory inplace.
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)
            

""" RL env runner """
from collections import defaultdict

import numpy as np


class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.state = {"latest_observation": self.env.reset()[0]}
        self._debug_iter = 1

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    def reset(self, **kwargs):
        """ Resets env and runner states. """
        self.state["latest_observation"], info = self.env.reset(**kwargs)
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.nsteps

        for i in range(self.nsteps):
            observations.append(self.state["latest_observation"])
            act = self.policy.act(self.state["latest_observation"])
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, terminated, truncated, _ = self.env.step(trajectory["actions"][-1])
            done = np.logical_or(terminated, truncated)
            self.state["latest_observation"] = obs
            rewards.append(rew)
            resets.append(done)
            self.step_var += self.nenvs or 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if not self.nenvs and np.all(done):
                self.state["env_steps"] = i + 1
                self.state["latest_observation"] = self.env.reset(seed=self._debug_iter)[0]
                self._debug_iter += 1

        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets)
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)
        return trajectory
    
    
import os
class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory):
        gamma = self.gamma
        lambda_ = self.lambda_

        # if trajectory['values'].ndim > 1:
        #     trajectory['values'] = trajectory['values'].squeeze(-1)

        n_steps = len(trajectory['observations'])
        deltas = []
        for i in range(n_steps):
            v_next = trajectory['values'][i + 1] if i < n_steps - 1 \
                else self.policy.act(trajectory["state"]["latest_observation"])['values']
            delta = trajectory['rewards'][i] - trajectory['values'][i]
            delta += (~trajectory['resets'][i]).astype(float) * self.gamma * v_next
            deltas.append(delta)
        deltas = np.array(deltas) # .squeeze(-1)

        advantages = []
        for t in range(n_steps):
            s = 0.0
            for l in range(n_steps - t):
                s += (self.gamma * self.lambda_)**l * float(deltas[t + l])
                if trajectory['resets'][t + l]:
                    break
            advantages.append(s)
        advantages = np.array(advantages)

        # powers = np.arange(n_steps) - np.maximum.accumulate(np.where(trajectory['resets'], np.arange(n_steps), 0), axis=-1)
        # gammalambda = np.power(self.gamma * self.lambda_, powers)
        # advantages = np.cumsum(gammalambda * deltas, axis=-1) / gammalambda

        trajectory['advantages'] = advantages
        trajectory['value_targets'] = advantages + trajectory['values']
        # trajectory['value_targets'] = np.expand_dims(advantages + trajectory['values'], axis=-1)

class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None

    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.

        Should be called at the beginning of each epoch.
        """
        trajectory_len = self.trajectory["observations"].shape[0]

        # permutation = np.random.permutation(trajectory_len)
        permutation = np.arange(trajectory_len)
        for key, value in self.trajectory.items():
            if key != 'state':
                self.trajectory[key] = value[permutation]

    def get_next(self):
        """ Returns next minibatch.  """
        if not self.trajectory:
            self.trajectory = self.runner.get_next()

        if self.minibatch_count == self.num_minibatches:
            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count += 1

        if self.epoch_count == self.num_epochs:
            self.trajectory = self.runner.get_next()

            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count = 0

        trajectory_len = self.trajectory["observations"].shape[0]

        batch_size = trajectory_len//self.num_minibatches

        minibatch = {}
        for key, value in self.trajectory.items():
            if key != 'state':
                minibatch[key] = value[self.minibatch_count*batch_size: (self.minibatch_count + 1)*batch_size]

        self.minibatch_count += 1

        for transform in self.transforms:
            transform(minibatch)

        return minibatch
    
class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory):
        mean = np.mean(trajectory['advantages'])
        std = np.std(trajectory['advantages'])
        trajectory['advantages'] = (trajectory['advantages'] - mean) / (std + 1e-7)
        
