import wandb
import click

from run_delivery_rl import main


@click.command()
@click.option('--sweep_id', type=str)
@click.option('--sweep_count', type=int)
def run(sweep_id, sweep_count):
    wandb.agent(sweep_id, function=main, count=sweep_count, project="delivery-RL-v2")


if __name__ == '__main__':
    run()
