import wandb
from run_reinforcement_training import main


if __name__ == '__main__':
    # wandb.login()
    sweep_id = 'dz6odfsb'

    wandb.agent(sweep_id, function=main, count=1, project="delivery-RL-v2")
