#########################
# DEPRECATED
#########################


# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# from collections import defaultdict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from itertools import accumulate
# from dispatch.dispatch import BaseDispatch, NeuralDispatch
# import numpy as np

# class Logger:
#     def __init__(self) -> None:
#         self.logs = defaultdict(list)
#         self.logs_mute = defaultdict(list)

#     def log(self, key, value, mute=False):
#         if not mute:
#             self.logs[key].append(value)
#         else:
#             self.logs_mute[key].append(value)

#     def plot(self, window_size=10, log_scale=False, start=0):

#         def moving_average(a, window_size) :
#             n = window_size
#             ret = np.cumsum(a, dtype=float)
#             ret[n:] = ret[n:] - ret[:-n]
#             ma = ret[n - 1:] / n
#             return np.append([ma[0]] * (n - 1), ma)
        
#         ncols = 3
#         nrows = (len(self.logs) - 1) // ncols + 1
#         figure, axis = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
#         for i, (k, v) in enumerate(self.logs.items()):
#             r = i // ncols
#             c = i % ncols
#             v_plot = v[start:]
#             if log_scale:
#                 v_plot = np.log(np.maximum(0.001, v[start:]))
#             if nrows != 1:
#                 axis[r, c].plot(v_plot, c='b')
#                 axis[r, c].plot(moving_average(v_plot, window_size), c='r')
#                 axis[r, c].set_title(k)
#             else:
#                 axis[c].plot(v, c='b')
#                 axis[c].plot(moving_average(v_plot, window_size), c='r')
#                 axis[c].set_title(k)
#         plt.show()


# class A2C:
#     def __init__(self, 
#                  sim, 
#                  dsp, 
#                  net: nn.Module, 
#                  opt: torch.optim.Optimizer, 
#                  n_steps=5, 
#                  n_sessions=2, 
#                  mode="CartPole-v1",
#                  value_loss_coef=1,
#                  regularization_loss_coef=1,
#         ) -> None:
#         self.net = net
#         self.opt = opt
#         self.n_steps = n_steps
#         self.n_sessions = n_sessions
#         self.gamma = 0.99
#         self.value_loss_coef = value_loss_coef
#         self.regularization_loss_coef = regularization_loss_coef
#         self.logger = Logger()
#         self.mode = mode

#         self.envs = [sim(dsp(self.net), mode) for _ in range(self.n_sessions)]

#     def train_step(self):
#         self.opt.zero_grad()
#         batch_of_sessions, batch_last_states = self.get_batch_of_sessions()
#         batch_dict = self.flatten_batch(batch_of_sessions)
#         self.compute_values_last_state(batch_dict, batch_last_states)
#         loss = self.compute_reinforce_loss(batch_dict)\
#               + self.compute_value_loss(batch_dict)\
#               + self.compute_regularization_loss(batch_dict)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.net.parameters(), 100)
#         self.logger.log('value grad norm', torch.square(batch_dict['values'].grad).mean())
#         self.logger.log('log_probs grad norm', torch.square(batch_dict['log_probs'].grad).mean())
#         self.logger.log('grad norm', self.compute_grad_norms())
#         self.logger.log('total loss', loss.item())
#         self.opt.step()

#         return loss.item()
    
#     def compute_grad_norms(self):
#         avg_grad_norms = []
#         for p in self.net.parameters():
#             if p.grad is None:
#                 continue
#             avg_grad_norms.append(torch.square(p.grad).mean().item())
#         return np.mean(avg_grad_norms)

#     def get_batch_of_sessions(self):
#         batch_sessions = []
#         batch_last_states = []
#         num_resets = 0
#         for env in self.envs:
#             session = []
#             for step in range(self.n_steps):
#                 state = env.GetState()
#                 env.Next()
#                 reward = env.GetReward()
#                 action = env.GetAction()
#                 session.append({
#                     'state': state,
#                     'reward': reward,
#                     'action': action,
#                     'step': step,
#                 })
#                 self.logger.log('step info', session[-1], mute=True)
#                 if self.mode == 'MountainCar-v0' and reward == 100:
#                     num_resets += 1
#             batch_last_states.append(env.GetState())
#             self.compute_cumulative_rewards(session)
#             batch_sessions.append(session)
        
#         self.logger.log('resets', num_resets)
#         return batch_sessions, batch_last_states

#     def compute_cumulative_rewards(self, session):
#         v_last = 0
#         for i in range(self.n_steps)[::-1]:
#             v_last = v_last * self.gamma + session[i]['reward']
#             session[i]['cum_reward'] = v_last

#     def flatten_batch(self, batch):
#         env_states = []
#         cum_rewards = []
#         actions = []
#         for session in batch:
#             for state in session:
#                 env_states.append(state['state'])
#                 cum_rewards.append(state['cum_reward'])
#                 actions.append(state['action'])
#         batch_dict = {}
#         batch_dict['cum_rewards'] = torch.tensor(cum_rewards, dtype=torch.float32)
#         self.run_network(batch_dict, env_states, actions)
#         self.logger.log('avg cum reward', np.mean(cum_rewards))
#         self.logger.log('prop actions', np.mean(actions))

#         return batch_dict
    
#     def run_network(self, batch_dict, env_states, actions):
#         logits, values = self.net(torch.tensor(np.array(env_states), dtype=torch.float32))
#         actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1)
#         probs = F.softmax(logits, dim=-1)
#         batch_dict['probs'] = torch.gather(probs, 1, actions).squeeze(-1)
#         log_probs = F.log_softmax(logits, dim=-1)
#         batch_dict['log_probs'] = torch.gather(log_probs, 1, actions).squeeze(-1)
#         batch_dict['values'] = values.squeeze(-1)

#         batch_dict['values'].retain_grad()
#         batch_dict['log_probs'].retain_grad()

#     def compute_values_last_state(self, batch_dict, batch_last_states):
#         _, last_state_values = self.net(torch.tensor(np.array(batch_last_states), dtype=torch.float32))
#         pows = torch.arange(self.n_steps, 0, -1)
#         last_values_discount = torch.pow(self.gamma, pows).unsqueeze(0) * last_state_values
#         batch_dict['cum_rewards'] += last_values_discount.flatten()

#     def compute_reinforce_loss(self, batch_dict):
#         log_probs = batch_dict['log_probs']
#         cum_rewards = batch_dict['cum_rewards'].detach()
#         values = batch_dict['values'].detach()
#         # values = 0
#         loss = -torch.mean(log_probs * (cum_rewards - values))
#         self.logger.log('policy_loss', loss.item())
#         return loss

#     def compute_value_loss(self, batch_dict):
#         cum_rewards = batch_dict['cum_rewards'].detach()
#         values = batch_dict['values']
#         # values = torch.tensor(0.0)
#         loss = F.mse_loss(values, cum_rewards)**0.5 * self.value_loss_coef
#         self.logger.log('value_loss', loss.item())
#         return loss

#     def compute_regularization_loss(self, batch_dict):
#         probs = batch_dict['probs']
#         log_probs = batch_dict['log_probs']
#         loss = torch.mean(log_probs * probs) * self.regularization_loss_coef
#         self.logger.log('reg_loss', loss.item())
#         return loss
    
#     def plot_logs(self, window_size=10, log_scale=False, start=0):
#         self.logger.plot(window_size, log_scale=log_scale, start=start)
