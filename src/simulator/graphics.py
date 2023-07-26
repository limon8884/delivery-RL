import matplotlib.pyplot as plt
import numpy as np


def plot_CR(metrics):
    print('micro average CR: ',
          np.mean([m['completed_orders'] / m['finished_orders'] for m in metrics if m['finished_orders'] != 0]))
    print('macro average CR: ',
          np.sum([m['completed_orders'] for m in metrics]) / np.sum([m['finished_orders'] for m in metrics]))
    plt.title('CR')
    plt.ylim(0)
    plt.plot([m['completed_orders'] / m['finished_orders'] for m in metrics if m['finished_orders'] != 0])
    plt.show()


def plot_counts(metrics):
    print('average free couriers: ', np.mean([m['current_free_couriers'] for m in metrics]))
    print('average free orders: ', np.mean([m['current_free_orders'] for m in metrics]))
    print('average active routes: ', np.mean([m['current_active_routes'] for m in metrics]))
    plt.title('nums')
    plt.plot([m['current_free_couriers'] for m in metrics], label='c')
    plt.plot([m['current_free_orders'] for m in metrics], label='o')
    plt.plot([m['current_active_routes'] for m in metrics], label='ar')
    plt.legend()
    plt.show()
