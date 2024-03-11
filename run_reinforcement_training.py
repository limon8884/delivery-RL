import json
import click
import uuid
import torch
import numpy
import random
import wandb
from tqdm import tqdm

from src.reinforcement.base import Runner, InferenceMetricsRunner
from src.reinforcement.delivery import DeliveryMaker
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.evaluation import evaluate


@click.command()
@click.option('--group_run', '-g', 'group_run', type=str)
@click.option('--description', '-d', 'description', type=str)
@click.option('--total_iters', required=False, type=int)
@click.option('--device', required=False, type=str)
@click.option('--use_wandb', required=False, type=bool)
@click.option('--n_envs', required=False, type=int)
@click.option('--trajectory_lenght', required=False, type=int)
@click.option('--batch_size', required=False, type=int)
@click.option('--num_epochs_per_traj', required=False, type=int)
@click.option('--max_num_points_in_route', required=False, type=int)
@click.option('--sampler_mode', required=False, type=str)
@click.option('--learning_rate', required=False, type=float)
@click.option('--optimizer', required=False, type=str)
@click.option('--rmsprop_alpha', required=False, type=float)
@click.option('--sgd_momentum', required=False, type=float)
@click.option('--scheduler_max_lr', required=False, type=float)
@click.option('--scheduler_pct_start', required=False, type=float)
@click.option('--exploration_temperature', required=False, type=float)
@click.option('--ppo_cliprange', required=False, type=float)
@click.option('--ppo_value_loss_coef', required=False, type=float)
@click.option('--max_grad_norm', required=False, type=float)
@click.option('--gae_gamma', required=False, type=float)
@click.option('--gae_lambda', required=False, type=float)
@click.option('--checkpoint_path', required=False, type=str, default='checkpoints/')
@click.option('--history_db_path', required=False, type=str, default='histories/')
@click.option('--simulator_cfg_path', required=False, type=str, default='configs/simulator.json')
@click.option('--network_cfg_path', required=False, type=str, default='configs/network.json')
@click.option('--training_cfg_path', required=False, type=str, default='configs/training.json')
@click.option('--debug_info_path', required=False, type=str, default='debug_info/')
@click.option('--eval_num_simulator_steps', required=False, type=int)
@click.option('--eval_n_envs', required=False, type=int)
@click.option('--eval_trajectory_lenght', required=False, type=int)
@click.option('--eval_epochs_frequency', required=False, type=int)
def make_kwargs(**kwargs):
    training_cfg_path = kwargs['training_cfg_path']
    with open(training_cfg_path) as f:
        cfg = dict(json.load(f))
    for k, v in kwargs.items():
        assert k in cfg.keys(), f'Not found argument {k} in config!'
        if v is None:
            continue
        cfg[k] = v

    train_id = str(uuid.uuid4().hex)
    cfg['train_id'] = train_id
    cfg['checkpoint_path'] += train_id + '.pt'
    cfg['history_db_path'] += train_id + '.db'
    cfg['debug_info_path'] += train_id + '.txt'
    return cfg


def run_ppo(**kwargs):
    if kwargs['use_wandb']:
        wandb.login()
        wandb.init(
            project="delivery-RL-v2",
            name=kwargs['train_id'],
            config=kwargs
        )

    maker = DeliveryMaker(**kwargs)

    eval_runner = Runner(environment=maker.environment, actor_critic=maker.actor_critic,
                         n_envs=kwargs['eval_n_envs'], trajectory_lenght=kwargs['eval_trajectory_lenght'])
    inference_logger = InferenceMetricsRunner(runner=eval_runner, logger=maker.logger)
    dsp = NeuralSequantialDispatch(actor_critic=maker.actor_critic,
                                   max_num_points_in_route=kwargs['max_num_points_in_route'])

    for iteration in tqdm(range(kwargs['total_iters'])):
        maker.actor_critic.train()
        sample = maker.sampler.sample()
        maker.ppo.step(sample)
        if iteration % kwargs['eval_epochs_frequency'] == 0:
            maker.actor_critic.eval()
            inference_logger()
            metrics = evaluate(dispatch=dsp, run_id=iteration, **kwargs)
            for k, v in metrics.items():
                maker.logger.log(k, v)
            if not kwargs['use_wandb']:
                maker.logger.plot(window_size=10)
            torch.save(maker.actor_critic.state_dict(), kwargs['checkpoint_path'])


if __name__ == '__main__':
    torch.manual_seed(seed=0)
    numpy.random.seed(seed=0)
    random.seed(0)
    kwargs = make_kwargs(standalone_mode=False)
    print('Start training with id ' + kwargs['train_id'])
    run_ppo(**kwargs)
