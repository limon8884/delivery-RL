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
@click.option('-i', '--total_iters', required=False, type=int)
@click.option('--device', required=False, type=str)
@click.option('-w', '--use_wandb', required=False, default=True, is_flag=True, type=bool)
@click.option('-z', '--fix_zero_seed', required=False, default=False, is_flag=True, type=bool)
@click.option('-t', '--use_train_logs', required=False, default=False, is_flag=True, type=bool)
@click.option('--mask_fake_crr', required=False, type=bool)
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
@click.option('--ppo_entropy_loss_coef', required=False, type=float)
@click.option('--max_grad_norm', required=False, type=float)
@click.option('--gae_gamma', required=False, type=float)
@click.option('--gae_lambda', required=False, type=float)
@click.option('--reward_norm_gamma', required=False, type=float)
@click.option('--reward_norm_cliprange', required=False, type=float)
@click.option('--coef_reward_assigned', required=False, type=float)
@click.option('--coef_reward_cancelled', required=False, type=float)
@click.option('--eval_num_simulator_steps', required=False, type=int)
@click.option('--eval_n_envs', required=False, type=int)
@click.option('--eval_trajectory_lenght', required=False, type=int)
@click.option('--eval_epochs_frequency', required=False, type=int)
@click.option('--sweep_id', type=str)
@click.option('--sweep_count', type=int)
def make_kwargs(**kwargs):
    with open('configs/paths.json') as f:
        paths = dict(json.load(f))
    with open(paths['training_cfg_path']) as f:
        cfg = dict(json.load(f))
    with open(paths['eval_cfg_path']) as f:
        cfg.update(json.load(f))
    cfg.update(paths)

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
    maker = DeliveryMaker(**kwargs)

    if not kwargs['use_train_logs']:
        maker.ppo.metric_logger = None
    eval_runner = Runner(environment=maker.environment.copy(), actor_critic=maker.actor_critic,
                         n_envs=kwargs['eval_n_envs'], trajectory_lenght=kwargs['eval_trajectory_lenght'])
    inference_logger = InferenceMetricsRunner(runner=eval_runner, metric_logger=maker.metric_logger)
    dsp = NeuralSequantialDispatch(actor_critic=maker.actor_critic,
                                   max_num_points_in_route=kwargs['max_num_points_in_route'])

    for iteration in tqdm(range(kwargs['total_iters'])):
        maker.actor_critic.train()
        sample = maker.sampler.sample()
        maker.ppo.step(sample)
        if (iteration + 1) % kwargs['eval_epochs_frequency'] == 0:
            maker.actor_critic.eval()
            metrics = evaluate(dispatch=dsp, run_id=iteration, **kwargs)
            for k, v in metrics.items():
                maker.metric_logger.log(k, v)
            inference_logger()
            if not kwargs['use_wandb']:
                maker.metric_logger.plot(window_size=10)
            torch.save(maker.actor_critic.state_dict(), kwargs['checkpoint_path'])


def main():
    kwargs = make_kwargs(standalone_mode=False)
    if kwargs['fix_zero_seed']:
        torch.manual_seed(seed=0)
        numpy.random.seed(seed=0)
        random.seed(0)
    if kwargs['use_wandb']:
        wandb.init(
            project="delivery-RL-v2",
            name=kwargs['train_id'],
            config=kwargs
        )
        for k in kwargs:
            kwargs[k] = getattr(wandb.config, k, kwargs[k])
    print('Start training with id ' + kwargs['train_id'])
    print('kwargs: ', kwargs)
    run_ppo(**kwargs)


if __name__ == '__main__':
    main()
