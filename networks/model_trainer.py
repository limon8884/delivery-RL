from utils import *
from dispatch.utils import *
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time

from objects.gamble_triple import GambleTriple, random_triple

class ModelTrainer:
    def __init__(self, models_info, n_epochs, n_iters, bounds) -> None:
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.bounds = bounds
        
        self.names = []
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.batch_sizes = []
        
        for model_info in models_info:
            self.names.append(model_info['name'])
            self.models.append(model_info['model'])
            self.optimizers.append(model_info['optimizer'])
            self.schedulers.append(model_info['scheduler'])
            self.batch_sizes.append(model_info['batch_size'])

        self.losses = [[] for _ in range(len(self.models))]
        self.times = [[] for _ in range(len(self.models))]

    def run(self):
        for epoch in range(self.n_epochs):
            for model_ind, model in enumerate(self.models):
                print('model ' + self.names[model_ind] + ' is running...')
                start_time = time.time()
                optimizer = self.optimizers[model_ind]
                scheduler = self.schedulers[model_ind]
                batch_size = self.batch_sizes[model_ind]

                rolling_loss = []
                # rolling_metrics = {k: [] for k in metrics.keys()}
                for iter in range(self.n_iters):
                    triples = [random_triple(self.bounds) for _ in range(batch_size)]

                    optimizer.zero_grad()
                    loss = get_loss_solve(model, triples)
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    rolling_loss.append(loss.item())

                # for k in metrics.keys():
                #     metrics[k].append(np.mean(rolling_metrics[k]))

                self.losses[model_ind].append(np.mean(rolling_loss))
                self.times[model_ind].append(time.time() - start_time)

            clear_output()
            plt.title('loss')
            for i in range(len(self.models)):
                plt.plot(self.losses[i], label=self.names[i])
                plt.legend()
            plt.show()
        