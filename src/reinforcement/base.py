import typing
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
    def step(self, action: Action) -> tuple[State, float, bool, dict[str, float]]:
        '''
        Performs step in the environment
        Input: an action to perform
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
        # self.old_probs_tensor: list[torch.FloatTensor] = []

        self.last_state: State = state
        self.last_state_value: typing.Optional[float] = None

    def append(self, state: State, action: Action, reward: float, done: bool,
               log_probs_chosen: float, value: float):
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
        # self.old_probs_tensor.append(old_probs_tens)


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
                 trajectory_lenght: int,
                 parallel: bool = False
                 ) -> None:
        self.environment = environment
        self.actor_critic = actor_critic
        self.n_envs = n_envs
        self.trajectory_lenght = trajectory_lenght
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
        for _ in range(self.trajectory_lenght):
            with torch.no_grad():
                self.actor_critic(states)
            actions = self.actor_critic.get_actions_list(best_actions=best_actions)
            log_probs_list = self.actor_critic.get_log_probs_list()
            values_list = self.actor_critic.get_values_list()
            # old_probs_tens = self.actor_critic.get_log_probs_tensor()
            new_states: list[State] = []
            total_info = defaultdict(float)
            for idx in range(self.n_envs):
                new_state, reward, done, info = self._environments[idx].step(actions[idx])
                self._trajectories[idx].append(states[idx], actions[idx], reward, done,
                                               log_probs_list[idx], values_list[idx])
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
    def _normalize(array: list[float]) -> np.ndarray:
        array = np.array(array)
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
        self.reset()

    def reset(self) -> None:
        self.lenght = 0
        self._advantages: typing.Optional[torch.FloatTensor] = None
        # self._target_values: typing.Optional[torch.FloatTensor] = None
        self._log_probs_chosen: typing.Optional[torch.FloatTensor] = None
        self._values: typing.Optional[torch.FloatTensor] = None
        self._states: typing.Optional[list[State]] = None
        self._actions_chosen: typing.Optional[torch.LongTensor] = None
        self._rewards: typing.Optional[np.ndarray] = None

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
        # old_probs = []
        self._states = []
        for traj in trajectories:
            advantages.extend(self.gae(traj))
            log_probs_chosen.extend(traj.log_probs_chosen)
            values.extend(traj.values)
            actions.extend([a.to_index() for a in traj.actions])
            self._states.extend(traj.states)
            rewards.extend(traj.rewards)
            # old_probs.extend(traj.old_probs_tensor)
        # self._actions_chosen = torch.LongTensor(actions).to(device=self.device)
        # self._advantages = torch.FloatTensor(advantages).to(device=self.device)
        # self._log_probs_chosen = torch.FloatTensor(log_probs_chosen).to(device=self.device)
        # self._values = torch.FloatTensor(values).to(device=self.device)
        self._actions_chosen = torch.tensor(actions, dtype=torch.int64).to(device=self.device)
        self._advantages = torch.tensor(advantages, dtype=torch.float).to(device=self.device)
        self._log_probs_chosen = torch.tensor(log_probs_chosen, dtype=torch.float).to(device=self.device)
        self._values = torch.tensor(values, dtype=torch.float).to(device=self.device)
        self._rewards = np.array(rewards)
        # self._old_probs_tens = old_probs

        self.lenght = len(self._advantages)
        self.shuffle()
        assert len(self._log_probs_chosen) == self.lenght
        assert len(self._values) == self.lenght
        assert len(self._states) == self.lenght
        assert len(self._actions_chosen) == self.lenght

    def shuffle(self) -> None:
        '''
        Shuffles trajectories in buffer
        '''
        self._iter = 0
        self._perm = torch.randperm(self.lenght, device=self.device)

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
            'advantages': self.normalize_advantages(self._advantages[choices]),
            'log_probs_chosen': self._log_probs_chosen[choices],
            'values': self._values[choices],
            'states': [self._states[i.item()] for i in choices],
            'actions_chosen': self._actions_chosen[choices],
            # 'old_probs': [self._old_probs_tens[i.item()] for i in choices],
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
        total_traj_len = self.runner.n_envs * self.runner.trajectory_lenght
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

    def plot(self, window_size=10, log_scale=False, start=0):
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
        total_reward = 0.0
        total_length = 0
        action_probs = []
        self.runner.actor_critic.eval()
        self.runner.reset()
        trajs = self.runner.run()
        for traj in trajs:
            for reward, reset, log_prob_chosen in zip(traj.rewards, traj.resets, traj.log_probs_chosen):
                total_length += 1
                total_reward += reward
                action_probs.append(np.exp(log_prob_chosen))
                if reset:
                    break

        self.metric_logger.log('avg episode reward', total_reward / self.runner.n_envs)
        self.metric_logger.log('avg episode length', total_length / self.runner.n_envs)
        self.metric_logger.log('avg step reward', total_reward / total_length)
        self.metric_logger.log('avg chosen prob', np.mean(action_probs))
        self.metric_logger.log('std chosen prob', np.std(action_probs))

        total_info = defaultdict(float)
        for info in self.runner._statistics:
            for k, v in info.items():
                total_info[k] += v
        for k, v in total_info.items():
            self.metric_logger.log('avg ' + k, v / self.runner.trajectory_lenght / self.runner.n_envs)

        self.metric_logger.commit(step=self.called_counter)
        self.called_counter += 1


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

    def _policy_loss(self, sample: dict[str, torch.FloatTensor | list[State]], new_log_probs: torch.FloatTensor):
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
            self.metric_logger.log('mean_log_prob_chosen', new_log_probs_chosen.mean().item())
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
        return -(torch.exp(new_log_probs) * new_log_probs).sum(dim=-1).mean()

    def _loss(self, sample: dict[str, torch.FloatTensor | list[State]]):
        self.actor_critic(sample['states'])
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
            self.metric_logger.commit(step=self._step)
            self._step += 1
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()


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
