import json
import click

from src.reinforcement.delivery import run_ppo


@click.command()
@click.argument('run_id', nargs=1)
@click.option('--n_envs', required=False, type=int)
@click.option('--trajectory_lenght', required=False, type=int)
@click.option('--eval_n_envs', required=False, type=int)
@click.option('--eval_trajectory_lenght', required=False, type=int)
@click.option('--batch_size', required=False, type=int)
@click.option('--num_epochs_per_traj', required=False, type=int)
@click.option('--total_iters', required=False, type=int)
@click.option('--eval_epochs_frequency', required=False, type=int)
@click.option('--max_num_points_in_route', required=False, type=int)
@click.option('--learning_rate', required=False, type=float)
@click.option('--sampler_mode', required=False, type=str)
@click.option('--device', required=False, type=str)
@click.option('--optimizer', required=False, type=str)
@click.option('--rmsprop_alpha', required=False, type=float)
@click.option('--sgd_momentum', required=False, type=float)
@click.option('--scheduler_max_lr', required=False, type=float)
@click.option('--scheduler_pct_start', required=False, type=float)
@click.option('--use_wandb', required=False, type=bool)
@click.option('--checkpoint_path', required=False, type=str, default='checkpoint.pt')
@click.option('--database_logger_path', required=False, type=str, default='history.db')
@click.option('--simulator_cfg_path', required=False, type=str, default='configs/simulator.json')
@click.option('--network_cfg_path', required=False, type=str, default='configs/network.json')
@click.option('--training_cfg_path', required=False, type=str, default='configs/training.json')
def get_kwargs(**kwargs):
    training_cfg_path = kwargs['training_cfg_path']
    with open(training_cfg_path) as f:
        cfg = dict(json.load(f))
    for k, v in kwargs.items():
        assert k in cfg.keys(), f'Not found argument {k} in config!'
        if v is None:
            continue
        cfg[k] = v
    return cfg


if __name__ == '__main__':
    cfg = get_kwargs(standalone_mode=False)
    run_ppo(**cfg)
