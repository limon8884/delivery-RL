import typing
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from IPython.display import clear_output
from collections import defaultdict


class Action:
    '''
    Base class for action in RL
    '''
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def to_index(self) -> int:
        '''
        Returns an index of the action
        '''
        raise NotImplementedError


class State:
    '''
    Base class for state in RL
    '''
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


class BaseEnvironment:
    '''
    A base class of environment in RL
    '''
    def step(self, action: Action, reset: bool = False) -> tuple[State, float, bool, dict[str, float]]:
        '''
        Performs step in the environment
        Input: 
        * action - an action to perform
        * reset - wheather it is the last action in the trajectory (done should be True)
        Returns: a tuple of
        * new_state
        * reward
        * done
        * info - additional information about the interaction
        '''
        raise NotImplementedError

    def reset(self, seed: typing.Optional[int] = None) -> State:
        '''
        Resets the environment
        Returns the initial state
        '''
        raise NotImplementedError

    def copy(self) -> 'BaseEnvironment':
        raise NotImplementedError


class Trajectory:
    '''
    A trajectory of RL agent
    stores all the information about interactions between agent and environment
    '''
    def __init__(self, state) -> None:
        self.lenght = 0
        self.states: list[State] = []
        self.actions: list[Action] = []
        self.rewards: list[float] = []
        self.resets: list[bool] = []
        self.log_probs_chosen: list[float] = []
        self.values: list[float] = []
        self.entropies: list[float] = []

        self.last_state: State = state
        self.last_state_value: typing.Optional[float] = None

    def append(self, state: State, action: Action, reward: float, done: bool,
               log_probs_chosen: float, value: float, entropy: float):
        '''
        Add an information about new step in trajectory
        '''
        self.lenght += 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.resets.append(done)
        self.log_probs_chosen.append(log_probs_chosen)
        self.values.append(value)
        self.entropies.append(entropy)


class BaseActorCritic(nn.Module):
    '''
    A base class of actor-critic model for agent decision making
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, state_list: list[State]) -> None:
        '''
        Makes a backward throw the network and stores the output tensors
        Input: a batch of states
        '''
        raise NotImplementedError

    def get_log_probs_list(self) -> list[float]:
        '''
        Should be called after `self.forward`
        Returns log probs of chosen actions for every item in the batch
        '''
        raise NotImplementedError

    def get_values_list(self) -> list[float]:
        '''
        Should be called after `self.forward`
        Returns values of new states for every item in the batch
        '''
        raise NotImplementedError

    def get_actions_list(self, best_actions=False) -> list[Action]:
        '''
        Should be called after `self.forward`
        Returns chosen actions for every item in the batch

        best_actions - if True the action with the highest probability will be sampled
        If False the action samples randomly
        '''
        raise NotImplementedError

    def get_values_tensor(self) -> torch.FloatTensor:
        '''
        Should be called after `self.forward` with gradient
        Returns the values tensor of shape [batch_size, ]
        '''
        raise NotImplementedError

    def get_log_probs_tensor(self) -> torch.FloatTensor:
        '''
        Should be called after `self.forward` with gradient
        Returns the log_probs tensor of all actions of shape [batch_size, action_dim]
        '''
        raise NotImplementedError


class Runner:
    '''
    A runner which runs a batch of environments and gather their trajectories
    '''
    def __init__(self,
                 environment: BaseEnvironment,
                 actor_critic: BaseActorCritic,
                 n_envs: int,
                 trajectory_length: int,
                 parallel: bool = False,
                 ) -> None:
        self.environment = environment.copy()
        self.actor_critic = actor_critic
        self.n_envs = n_envs
        self.trajectory_length = trajectory_length
        self.parallel = parallel
        self.reset()

    def reset(self, seeds: typing.Optional[list[int]] = None) -> None:
        self._environments = [self.environment.copy() for _ in range(self.n_envs)]
        if seeds is None:
            self._trajectories = [Trajectory(env.reset()) for env in self._environments]
        else:
            self._trajectories = [Trajectory(env.reset(seed)) for env, seed in zip(self._environments, seeds)]
        self._statistics: list[dict[str, float]] = []

    def run(self, best_actions=False) -> list[Trajectory]:
        states = [traj.last_state for traj in self._trajectories]
        for trajectory_step in range(self.trajectory_length):
            is_last_step = trajectory_step == self.trajectory_length - 1
            with torch.no_grad():
                self.actor_critic(states)
            actions = self.actor_critic.get_actions_list(best_actions=best_actions)
            log_probs_list = self.actor_critic.get_log_probs_list()
            values_list = self.actor_critic.get_values_list()
            log_probs = self.actor_critic.get_log_probs_tensor()
            entropies = (-(torch.exp(log_probs) * log_probs).sum(dim=-1)).tolist()
            new_states: list[State] = []
            total_info: dict[str, float] = defaultdict(float)
            for idx in range(self.n_envs):
                new_state, reward, done, info = self._environments[idx].step(actions[idx], reset=is_last_step)
                self._trajectories[idx].append(states[idx], actions[idx], reward, done,
                                               log_probs_list[idx], values_list[idx], entropies[idx])
                new_states.append(new_state)
                for k, v in info.items():
                    total_info[k] += v
            states = new_states
            self._statistics.append(total_info)
        with torch.no_grad():
            self.actor_critic(states)
        for state, value, trajectory in zip(states, self.actor_critic.get_values_list(), self._trajectories):
            trajectory.last_state = state
            trajectory.last_state_value = value
        return self._trajectories


class GAE:
    '''Generalized Advantage Estimator class'''
    def __init__(self, gamma=0.99, lambda_=0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory: Trajectory) -> list[float]:
        '''
        Computes GAE for the trajectory
        Returns trajectoty advantages
        '''
        rewards = trajectory.rewards
        n_steps = trajectory.lenght
        deltas = []
        for t in range(n_steps):
            v_next = trajectory.values[t + 1] if t < n_steps - 1 else trajectory.last_state_value
            delta = rewards[t] - trajectory.values[t]
            delta += float(not trajectory.resets[t]) * self.gamma * v_next
            deltas.append(delta)

        advantages = []
        for t in range(n_steps):
            s = 0.0
            for i in range(n_steps - t):
                s += (self.gamma * self.lambda_)**i * float(deltas[t + i])
                if trajectory.resets[t + i]:
                    break
            advantages.append(s)

        return advantages

    @staticmethod
    def _normalize(data: list[float]) -> np.ndarray:
        array = np.array(data)
        return (array - np.mean(array)) / (np.std(array) + 1e-7)


class RewardNormalizer:
    '''
    Normalize rewards inplace
    '''
    def __init__(self, gamma: float = 0.99, cliprange: float = 10.0) -> None:
        self.gamma = gamma
        self.cliprange = cliprange

    def __call__(self, trajectories: list[Trajectory]) -> None:
        rewards = np.array([traj.rewards for traj in trajectories])
        resets = np.array([traj.resets for traj in trajectories])
        for traj in trajectories:
            traj.rewards = []

        mvg_mean = 0.0
        mvg_var = 1.0
        mvg_value = np.zeros(rewards.shape[0])
        for i in range(rewards.shape[1]):
            mvg_value = self.gamma * mvg_value + rewards[:, i]
            delta = np.mean(mvg_value) - mvg_mean
            mvg_mean += delta / (i + 1)
            mvg_var = mvg_var * i / (i + 1) + np.var(mvg_value) / (i + 1) + delta**2 * i / (i + 1)**2
            result_vals = np.clip(rewards[:, i] / (mvg_var + 1e-7)**0.5, -self.cliprange, self.cliprange)
            mvg_value[resets[:, i]] = 0.0
            for traj, result_val in zip(trajectories, result_vals):
                traj.rewards.append(result_val)


class Buffer:
    '''
    A buffer for storing the trajectories and sampling their pieces for training
    '''
    def __init__(self, gae: GAE, device,
                 reward_normalizer: typing.Optional[RewardNormalizer] = None, replace=True) -> None:
        self.gae = gae
        self.reward_normalizer = reward_normalizer
        self.device = device
        self.replace = replace
        self.lenght = 0

    def update(self, trajectories: list[Trajectory]) -> None:
        '''
        Updates buffer with new trajectories
        '''
        if self.reward_normalizer is not None:
            self.reward_normalizer(trajectories)
        advantages = []
        log_probs_chosen = []
        values = []
        actions = []
        rewards = []
        resets = []
        self._states = []
        for traj in trajectories:
            advantages.extend(self.gae(traj))
            log_probs_chosen.extend(traj.log_probs_chosen)
            values.extend(traj.values)
            actions.extend([a.to_index() for a in traj.actions])
            self._states.extend(traj.states)
            rewards.extend(traj.rewards)
            resets.extend(traj.resets)
        self._actions_chosen = torch.tensor(actions, dtype=torch.int64).to('cpu')
        self._advantages = torch.tensor(advantages, dtype=torch.float).to('cpu')
        self._log_probs_chosen = torch.tensor(log_probs_chosen, dtype=torch.float).to('cpu')
        self._values = torch.tensor(values, dtype=torch.float).to('cpu')
        self._rewards = np.array(rewards)
        self._resets = torch.tensor(resets, dtype=torch.bool).to('cpu')

        self.lenght = len(self._advantages)
        self.shuffle()
        assert len(self._log_probs_chosen) == self.lenght
        assert len(self._values) == self.lenght
        assert len(self._states) == self.lenght
        assert len(self._actions_chosen) == self.lenght
        assert len(self._resets) == self.lenght

    def shuffle(self) -> None:
        '''
        Shuffles trajectories in buffer
        '''
        self._iter = 0
        self._perm = torch.randperm(self.lenght, device='cpu')

    def sample(self, size: int) -> dict[str, torch.FloatTensor | list[State]]:
        '''
        Samples a batch of cells of trajectory
        Input: size - a size of batch
        '''
        assert self.lenght > 0
        assert self._iter + size <= self.lenght, 'can not sample, buffer ended('
        choices = self._perm[self._iter: self._iter + size]
        self._iter += size
        return {
            'advantages': self.normalize_advantages(self._advantages[choices]).to(self.device),
            'log_probs_chosen': self._log_probs_chosen[choices].to(self.device),
            'values': self._values[choices].to(self.device),
            'states': [self._states[i.item()] for i in choices],
            'actions_chosen': self._actions_chosen[choices].to(self.device),
            'resets': self._resets[choices].to(self.device),
        }

    @staticmethod
    def normalize_advantages(adv: torch.FloatTensor) -> torch.FloatTensor:
        mean = torch.mean(adv)
        std = torch.std(adv)
        return (adv - mean) / (std + 1e-7)


class TrajectorySampler:
    def __init__(self, runner: Runner, buffer: Buffer, num_epochs_per_traj: int, batch_size: int) -> None:
        self.runner = runner
        self.buffer = buffer
        self.num_epochs_per_traj = num_epochs_per_traj
        self.batch_size = batch_size
        self._iter = 0

    def sample(self) -> dict[str, torch.FloatTensor | list[State]]:
        total_traj_len = self.runner.n_envs * self.runner.trajectory_length
        if self._iter % (total_traj_len * self.num_epochs_per_traj // self.batch_size) == 0:
            self.runner.reset()
            trajs = self.runner.run()
            self.buffer.update(trajs)
        if self._iter % (total_traj_len // self.batch_size) == 0:
            self.buffer.shuffle()
        self._iter += 1
        return self.buffer.sample(size=self.batch_size)


class MetricLogger:
    def __init__(self, use_wandb: bool = False) -> None:
        self.logs = defaultdict(list)
        self.logs_mute = defaultdict(list)
        self.wandb_logs = {} if use_wandb else None

    def log(self, key, value, mute=False):
        if self.wandb_logs is not None:
            self.wandb_logs[key] = value
            return

        if not mute:
            self.logs[key].append(value)
        else:
            self.logs_mute[key].append(value)

    def commit(self, step: int):
        if self.wandb_logs is not None:
            wandb.log({**self.wandb_logs, 'iter': step})
            self.wandb_logs = {}

    def plot(self, window_size=1, log_scale=False, start=0):
        def moving_average(a, window_size):
            n = window_size
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            ma = ret[n - 1:] / n
            return np.append([ma[0]] * (n - 1), ma)

        clear_output()
        ncols = 3
        nrows = (len(self.logs) - 1) // ncols + 1
        figure, axis = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
        for i, (k, v) in enumerate(self.logs.items()):
            if len(v) < window_size:
                continue
            r = i // ncols
            c = i % ncols
            v_plot = v[start:]
            if log_scale:
                v_plot = np.log(np.maximum(0.001, v[start:]))
            if nrows != 1:
                axis[r, c].plot(v_plot, c='b')
                axis[r, c].plot(moving_average(v_plot, window_size), c='r')
                axis[r, c].set_title(k)
            else:
                axis[c].plot(v, c='b')
                axis[c].plot(moving_average(v_plot, window_size), c='r')
                axis[c].set_title(k)
        plt.show()


class InferenceMetricsRunner:
    def __init__(self, runner: Runner, metric_logger: MetricLogger) -> None:
        self.runner = runner
        self.metric_logger = metric_logger
        self.called_counter = 0

    def __call__(self) -> None:
        self.runner.actor_critic.eval()
        self.runner.reset()
        trajs = self.runner.run(best_actions=True)
        for k, v in self.get_metrics_from_trajectory(trajs).items():
            self.metric_logger.log(k, v)

        total_info = defaultdict(float)
        for info in self.runner._statistics:
            for k, v in info.items():
                total_info[k] += v
        for k, v in total_info.items():
            self.metric_logger.log('SIM: ' + k, v / self.runner.trajectory_length / self.runner.n_envs)

        self.metric_logger.commit(step=self.called_counter)
        self.called_counter += 1

    @staticmethod
    def get_metrics_from_trajectory(trajs: list[Trajectory]) -> dict[str, float]:
        raise NotImplementedError


class PPO:
    def __init__(self,
                 actor_critic: BaseActorCritic,
                 opt: torch.optim.Optimizer,
                 scheduler: typing.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 metric_logger: typing.Optional[MetricLogger] = None,
                 **kwargs,
                 ):
        self.metric_logger = metric_logger
        self.actor_critic = actor_critic
        self.opt = opt
        self.scheduler = scheduler
        self.cliprange = kwargs['ppo_cliprange']
        self.value_loss_coef = kwargs['ppo_value_loss_coef']
        self.entropy_loss_coef = kwargs['ppo_entropy_loss_coef']
        self.max_grad_norm = kwargs['max_grad_norm']
        self.device = kwargs['device']
        self.debug_file_path = kwargs['debug_info_path']
        self._step = 0

    def _policy_loss(self, sample: dict[str, torch.Tensor | list[State]], new_log_probs: torch.Tensor):
        """ Computes and returns policy loss on a given trajectory. """
        a = sample['advantages']
        old_log_probs_chosen = sample['log_probs_chosen']
        actions_chosen = sample['actions_chosen']

        new_log_probs_chosen = torch.gather(
            new_log_probs, dim=-1, index=actions_chosen.unsqueeze(-1)
        ).squeeze(-1)
        r = torch.exp(new_log_probs_chosen - old_log_probs_chosen)
        loss = -torch.minimum(r * a, torch.clamp(r, 1 - self.cliprange, 1 + self.cliprange) * a).mean()
        if self.metric_logger:
            self.metric_logger.log('policy_loss', loss.item())
            self.metric_logger.log('new_mean_prob_chosen', new_log_probs_chosen.exp().mean().item())
            self.metric_logger.log('old_mean_prob_chosen', old_log_probs_chosen.exp().mean().item())
        return loss

    def _value_loss(self, sample: dict[str, torch.FloatTensor | list[State]], new_values: torch.FloatTensor):
        """ Computes and returns value loss on a given trajectory. """
        old_values = sample['values']
        target_values = sample['values'] + sample['advantages']
        l_simple = torch.square(new_values - target_values)
        l_clipped = torch.square(
            old_values + torch.clamp(new_values - old_values, -self.cliprange, self.cliprange) - target_values
        )
        loss = torch.maximum(l_simple, l_clipped).mean()
        if self.metric_logger:
            self.metric_logger.log('value_loss', loss.item())
            self.metric_logger.log('mean_value', new_values.mean().item())
        return loss

    def _entropy_loss(self, new_log_probs: torch.FloatTensor):
        return (torch.exp(new_log_probs) * new_log_probs).sum(dim=-1).mean()

    def _loss(self, sample: dict[str, torch.Tensor | list[State]]):
        self.actor_critic(sample['states'])

        if self.metric_logger:
            log_probs = self.actor_critic.get_log_probs_tensor()
            self.metric_logger.log('entropy', -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean().item())
            # is_last_action = sum([(state.last() == act_idx.item() and state.has_) for act_idx, state
            #                      in zip(sample['actions_chosen'], sample['states'])]) / len(sample['states'])
            # empty_action = sum([(state.size() == 0) for state in sample['states']]) / len(sample['states'])
            # self.metric_logger.log('last action', is_last_action)
            # self.metric_logger.log('empty action', empty_action)

        policy_loss = self._policy_loss(sample, self.actor_critic.get_log_probs_tensor())
        value_loss = self._value_loss(sample, self.actor_critic.get_values_tensor())
        entropy_loss = self._entropy_loss(self.actor_critic.get_log_probs_tensor())

        return policy_loss + self.value_loss_coef * value_loss + self.entropy_loss_coef * entropy_loss

    def step(self, sample):
        """
        Computes the loss function and performs a single gradient step
        """
        self.opt.zero_grad()
        self.actor_critic.train()
        loss = self._loss(sample)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        if self.metric_logger:
            self.metric_logger.log('grad norm', grad_norm.item())
            self.metric_logger.log('total loss', loss.item())
            self.metric_logger.log('has reset', int(sample['resets'].any().item()))
            if self.scheduler is not None:
                self.metric_logger.log('lr', self.scheduler.get_last_lr()[0])
            self.metric_logger.commit(step=self._step)
            self._step += 1
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()


class CloningPPO(PPO):
    def __init__(self, actor_critic: BaseActorCritic, opt: Optimizer, scheduler: _LRScheduler | None = None,
                 metric_logger: MetricLogger | None = None, **kwargs):
        super().__init__(actor_critic, opt, scheduler, metric_logger, **kwargs)
        self.value_loss_coef = 0.0
        self.entropy_loss_coef = 0.0

    def _policy_loss(self, sample: dict[str, torch.Tensor | list[State]], new_log_probs: torch.Tensor):
        actions_chosen = sample['actions_chosen']
        new_log_probs_chosen = torch.gather(
            new_log_probs, dim=-1, index=actions_chosen.unsqueeze(-1)
        ).squeeze(-1)
        loss = -new_log_probs_chosen.mean()
        if self.metric_logger:
            self.metric_logger.log('policy_loss', loss.item())
            self.metric_logger.log('new_mean_prob_chosen', new_log_probs_chosen.exp().mean().item())
        return loss


class BaseMaker:
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError

    @property
    def ppo(self) -> PPO:
        raise NotImplementedError

    @property
    def sampler(self) -> TrajectorySampler:
        raise NotImplementedError

    @property
    def actor_critic(self) -> BaseActorCritic:
        raise NotImplementedError

    @property
    def environment(self) -> BaseEnvironment:
        raise NotImplementedError

    @property
    def metric_logger(self) -> MetricLogger:
        raise NotImplementedError


def make_optimizer(parameters: typing.Iterable, **kwargs):
    optimizer = kwargs['optimizer']
    assert optimizer in ['adam', 'rmsprop', 'sgd'], f'Optimizer {optimizer} is not available'
    if optimizer == 'adam':
        return torch.optim.AdamW(parameters, lr=kwargs['learning_rate'])
    if optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=kwargs['learning_rate'], alpha=kwargs['rmsprop_alpha'])
    if optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr=kwargs['learning_rate'], momentum=kwargs['sgd_momentum'])
