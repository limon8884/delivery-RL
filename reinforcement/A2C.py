import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import defaultdict
import torch
import torch.nn as nn


class A2C:
    def __init__(self, sim, net: nn.Module, opt: torch.optim.Optimizer, n_steps=5, n_sessions=2) -> None:
        self.net = net
        self.opt = opt
        self.sim = sim
        self.n_steps = n_steps
        self.n_sessions = n_sessions

        self.envs = [self.sim() for _ in range(self.n_sessions)]

    def train_step(self):
        batch_of_sessions = self.get_batch_of_sessions()
        self.compute_statistics(batch_of_sessions)
        batch_flatten_dict = self.reshape_batch(batch_of_sessions)
        loss = self.compute_reinforce_loss(batch_flatten_dict)\
              + self.compute_value_loss(batch_flatten_dict)\
              + self.compute_regularization_loss(batch_flatten_dict)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def get_batch_of_sessions(self):
        batch = []
        for env in self.envs:
            session = []
            for step in range(self.n_steps):
                session.append(env.Getstate())
                env.Next()
            batch.append(session)
        return batch

    def compute_statistics(self, batch_of_sessions):
        pass

    def get_state_dict(self, state):
        pass

    def reshape_batch(batch):
        pass

    def get_value_scores(self):
        pass

    def compute_reinforce_loss(self, batch_dict):
        log_probs = batch_dict['log_probs']
        cum_rewards = batch_dict['cum_rewards']
        values = batch_dict['values']

    def compute_value_loss(self, batch_dict):
        cum_rewards = batch_dict['cum_rewards'].detach()
        values = batch_dict['values']

    def compute_regularization_loss(self, batch_dict):
        probs = batch_dict['probs']
        log_probs = batch_dict['log_probs']











class A2C_old:
    def __init__(self,
                #  policy,
                 runner,
                 optimizer,
                 scheduler=None,
                 value_loss_coef=0.25,
                 entropy_coef=0.01,
                 max_grad_norm=0.5):
        # self.policy = policy
        self.runner = runner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.init_logs()
        self.smoothed_logs = defaultdict(list)

    def init_logs(self):
        self.logs = {}
        self.logs['entropy'] = []
        self.logs['value_loss'] = []
        self.logs['policy_loss'] = []
        self.logs['value_targets'] = []
        self.logs['value_preds'] = []
        self.logs['grad_norm'] = []
        self.logs['advantages'] = []
        self.logs['total_loss'] = []
        self.logs['rewards'] = []

    def policy_loss(self, trajectory):
        # You will need to compute advantages here.
        # <TODO: implement>
        value_targets = np.mean(trajectory['value_targets'].detach().numpy())
        value_preds = np.mean(trajectory['values'].detach().numpy())
        anvantage = value_targets - value_preds
        self.logs['value_targets'].append(value_targets)
        self.logs['value_preds'].append(value_preds)
        self.logs['advantages'].append(anvantage)

        value = -torch.sum( trajectory['log_probs'] * ( trajectory['value_targets'] - trajectory['values']).detach() )
        self.logs['policy_loss'].append(value.item())
        return value

    def value_loss(self, trajectory):
        # <TODO: implement>
        # value = nn.functional.mse_loss(trajectory['values'].float(), trajectory['value_targets'].float())
        value = nn.functional.mse_loss(trajectory['values'], trajectory['value_targets'].float())
        self.logs['value_loss'].append(value.item())
        return value
    
    def entropy_loss(self, trajectory):
        logits = trajectory['logits']
        probs = nn.functional.softmax(logits, dim=-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        value = torch.sum(probs * log_probs)
        self.logs['entropy'].append(value.item())
        return value

    def loss(self, trajectory):
        # <TODO: implement>
        value = self.policy_loss(trajectory) + self.value_loss_coef * self.value_loss(trajectory) + self.entropy_coef * self.entropy_loss(trajectory)
        self.logs['total_loss'].append(value.item())
        return value

    def step(self, trajectory):
        # <TODO: implement>
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.runner.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        total_norm = 0
        for p in self.runner.policy.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        self.logs['grad_norm'].append(total_norm)
        # print(total_norm)

    def update_smoothed_logs(self, period):
        l = len(self.smoothed_logs['total_loss']) * period
        for k, v in self.logs.items():
            new_value = np.mean(v[l:])
            self.smoothed_logs[k].append(new_value)
        
    def plot_logs(self, smoothed=True):
        clear_output()
        figure, axis = plt.subplots(3, 3, figsize=(20, 10))
        if smoothed:
            logs = self.smoothed_logs
        else:
            logs = self.logs
        for i, (k, v) in enumerate(logs.items()):
            r = i // 3
            c = i % 3
            axis[r, c].plot(v)
            axis[r, c].set_title(k)
        plt.show()
        