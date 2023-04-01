import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from dispatch.dispatch import BaseDispatch, NeuralDispatch
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = defaultdict(list)

    def log(self, key, value):
        self.logs[key].append(value)

    def plot(self):
        ncols = 3
        nrows = (len(self.logs) - 1) // ncols + 1
        figure, axis = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
        for i, (k, v) in enumerate(self.logs.items()):
            r = i // ncols
            c = i % ncols
            if nrows != 1:
                axis[r, c].plot(v)
                axis[r, c].set_title(k)
            else:
                axis[c].plot(v)
                axis[c].set_title(k)
        plt.show()


class A2C:
    def __init__(self, sim, dsp, net: nn.Module, opt: torch.optim.Optimizer, n_steps=5, n_sessions=2) -> None:
        self.net = net
        self.opt = opt
        self.n_steps = n_steps
        self.n_sessions = n_sessions
        self.gamma = 0.99
        self.logger = Logger()

        self.envs = [sim(dsp(self.net)) for _ in range(self.n_sessions)]

    def train_step(self):
        batch_of_sessions, batch_last_states = self.get_batch_of_sessions()
        batch_dict = self.flatten_batch(batch_of_sessions)
        self.compute_values_last_state(batch_dict, batch_last_states)
        loss = self.compute_reinforce_loss(batch_dict)\
              + self.compute_value_loss(batch_dict)\
              + self.compute_regularization_loss(batch_dict)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def get_batch_of_sessions(self):
        batch_sessions = []
        batch_last_states = []
        for env in self.envs:
            session = []
            for step in range(self.n_steps):
                state = env.GetState()
                env.Next()
                reward = env.GetReward()
                session.append({
                    'state': state,
                    'reward': reward,
                    'step': step,
                })
            batch_last_states.append(env.GetState())
            self.compute_cumulative_rewards(session)
            batch_sessions.append(session)
        return batch_sessions, batch_last_states

    def compute_cumulative_rewards(self, session):
        v_last = 0
        for i in range(self.n_steps)[::-1]:
            v_last = v_last * self.gamma + session[i]['reward']
            session[i]['cum_reward'] = v_last

    def flatten_batch(self, batch):
        env_states = []
        cum_rewards = []
        for session in batch:
            for state in session:
                env_states.append(state['state'])
                cum_rewards.append(state['cum_reward'])
        
        batch_dict = self.run_network(torch.tensor(env_states, dtype=torch.float32))
        batch_dict['cum_rewards'] = torch.tensor(cum_rewards, dtype=torch.float32)
        self.logger.log('avg cum reward', np.mean(cum_rewards))

        return batch_dict
    
    def run_network(self, input):
        logits, values = self.net(input)
        bacth_dict = {}
        bacth_dict['probs'] = F.softmax(logits, dim=-1)
        bacth_dict['log_probs'] = F.log_softmax(logits, dim=-1)
        bacth_dict['values'] = values.squeeze(-1)

        return bacth_dict

    def compute_values_last_state(self, batch_dict, batch_last_states):
        with torch.no_grad():
            _, last_state_values = self.net(torch.tensor(batch_last_states, dtype=torch.float32))
        pows = torch.arange(self.n_steps, 0, -1)
        last_values_discount = torch.pow(self.gamma, pows).unsqueeze(0) * last_state_values
        batch_dict['cum_rewards'] += last_values_discount.flatten()

    def compute_reinforce_loss(self, batch_dict):
        log_probs = batch_dict['log_probs']
        cum_rewards = batch_dict['cum_rewards']
        values = batch_dict['values']
        print(log_probs.shape, cum_rewards.shape, values.shape)
        loss = -torch.mean(log_probs * (cum_rewards - values))
        self.logger.log('policy_loss', loss.item())
        return loss

    def compute_value_loss(self, batch_dict):
        cum_rewards = batch_dict['cum_rewards'].detach()
        values = batch_dict['values']
        loss = F.mse_loss(values, cum_rewards)
        self.logger.log('value_loss', loss.item())
        return loss

    def compute_regularization_loss(self, batch_dict):
        probs = batch_dict['probs']
        log_probs = batch_dict['log_probs']
        loss = torch.mean(log_probs * probs)
        self.logger.log('reg_loss', loss.item())
        return loss
    
    def plot_logs(self):
        self.logger.plot()





# class A2C_old:
#     def __init__(self,
#                 #  policy,
#                  runner,
#                  optimizer,
#                  scheduler=None,
#                  value_loss_coef=0.25,
#                  entropy_coef=0.01,
#                  max_grad_norm=0.5):
#         # self.policy = policy
#         self.runner = runner
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#         self.max_grad_norm = max_grad_norm
#         self.init_logs()
#         self.smoothed_logs = defaultdict(list)

#     def init_logs(self):
#         self.logs = {}
#         self.logs['entropy'] = []
#         self.logs['value_loss'] = []
#         self.logs['policy_loss'] = []
#         self.logs['value_targets'] = []
#         self.logs['value_preds'] = []
#         self.logs['grad_norm'] = []
#         self.logs['advantages'] = []
#         self.logs['total_loss'] = []
#         self.logs['rewards'] = []

#     def policy_loss(self, trajectory):
#         # You will need to compute advantages here.
#         # <TODO: implement>
#         value_targets = np.mean(trajectory['value_targets'].detach().numpy())
#         value_preds = np.mean(trajectory['values'].detach().numpy())
#         anvantage = value_targets - value_preds
#         self.logs['value_targets'].append(value_targets)
#         self.logs['value_preds'].append(value_preds)
#         self.logs['advantages'].append(anvantage)

#         value = -torch.sum( trajectory['log_probs'] * ( trajectory['value_targets'] - trajectory['values']).detach() )
#         self.logs['policy_loss'].append(value.item())
#         return value

#     def value_loss(self, trajectory):
#         # <TODO: implement>
#         # value = nn.functional.mse_loss(trajectory['values'].float(), trajectory['value_targets'].float())
#         value = nn.functional.mse_loss(trajectory['values'], trajectory['value_targets'].float())
#         self.logs['value_loss'].append(value.item())
#         return value
    
#     def entropy_loss(self, trajectory):
#         logits = trajectory['logits']
#         probs = nn.functional.softmax(logits, dim=-1)
#         log_probs = nn.functional.log_softmax(logits, dim=-1)
#         value = torch.sum(probs * log_probs)
#         self.logs['entropy'].append(value.item())
#         return value

#     def loss(self, trajectory):
#         # <TODO: implement>
#         value = self.policy_loss(trajectory) + self.value_loss_coef * self.value_loss(trajectory) + self.entropy_coef * self.entropy_loss(trajectory)
#         self.logs['total_loss'].append(value.item())
#         return value

#     def step(self, trajectory):
#         # <TODO: implement>
#         self.optimizer.zero_grad()
#         loss = self.loss(trajectory)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm(self.runner.policy.model.parameters(), self.max_grad_norm)
#         self.optimizer.step()
#         if self.scheduler is not None:
#             self.scheduler.step()

#         total_norm = 0
#         for p in self.runner.policy.model.parameters():
#             param_norm = p.grad.data.norm(2)
#             total_norm += param_norm.item()**2
#         total_norm = total_norm**0.5
#         self.logs['grad_norm'].append(total_norm)
#         # print(total_norm)

#     def update_smoothed_logs(self, period):
#         l = len(self.smoothed_logs['total_loss']) * period
#         for k, v in self.logs.items():
#             new_value = np.mean(v[l:])
#             self.smoothed_logs[k].append(new_value)
        
#     def plot_logs(self, smoothed=True):
#         clear_output()
#         figure, axis = plt.subplots(3, 3, figsize=(20, 10))
#         if smoothed:
#             logs = self.smoothed_logs
#         else:
#             logs = self.logs
#         for i, (k, v) in enumerate(logs.items()):
#             r = i // 3
#             c = i % 3
#             axis[r, c].plot(v)
#             axis[r, c].set_title(k)
#         plt.show()
        