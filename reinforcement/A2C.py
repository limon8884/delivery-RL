import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import defaultdict

class A2C:
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
        