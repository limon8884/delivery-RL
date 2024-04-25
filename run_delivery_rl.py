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
from src.reinforcement.delivery2 import DeliveryMaker2
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.evaluation import evaluate


@click.command()
@click.option('--group_run', '-g', 'group_run', type=str)
@click.option('--description', '-d', 'description', type=str)
@click.option('--load_checkpoint', '-l', 'load_checkpoint', type=str, default="")
@click.option('-i', '--total_iters', required=False, type=int, default=100_000)
@click.option('-d', '--device', required=False, type=str, default='cuda')
@click.option('-w', '--use_wandb', required=False, default=True, is_flag=True, type=bool)
@click.option('-z', '--fix_zero_seed', required=False, default=False, is_flag=True, type=bool)
@click.option('-t', '--use_train_logs', required=False, default=False, is_flag=True, type=bool)
@click.option('-a', '--use_attn', required=False, default=False, is_flag=True, type=bool)
@click.option('--mode', required=False, default='v1', type=str)
@click.option('--use_pretrained_encoders', required=False, type=bool, default=True)
@click.option('--mask_fake_crr', required=False, type=bool, default=False)
@click.option('--use_dist', required=False, type=bool, default=True)
@click.option('--use_masks', required=False, type=bool, default=False)
@click.option('--n_envs', required=False, type=int, default=1)
@click.option('-T', '--trajectory_length', required=False, type=int, default=12_000)
@click.option('--batch_size', required=False, type=int, default=64)
@click.option('--num_epochs_per_traj', required=False, type=int, default=10)
@click.option('--max_num_points_in_route', required=False, type=int, default=2)
@click.option('--sampler_mode', required=False, type=str, default='dummy_sampler')
@click.option('--model_size', required=False, type=str, default='medium')
@click.option('--dropout', required=False, type=float, default=0.0)
@click.option('--learning_rate', required=False, type=float, default=1e-5)
@click.option('--optimizer', required=False, type=str, default='adam')
@click.option('--rmsprop_alpha', required=False, type=float, default=0.9)
@click.option('--sgd_momentum', required=False, type=float, default=0.9)
@click.option('--scheduler_max_lr', required=False, type=float, default=3e-4)
@click.option('--scheduler_pct_start', required=False, type=float, default=0.3)
@click.option('--exploration_temperature', required=False, type=float, default=1.0)
@click.option('--ppo_cliprange', required=False, type=float, default=0.2)
@click.option('--ppo_value_loss_coef', required=False, type=float, default=0.25)
@click.option('--ppo_entropy_loss_coef', required=False, type=float, default=0.0)
@click.option('--max_grad_norm', required=False, type=float, default=1.0)
@click.option('--gae_gamma', required=False, type=float, default=0.99)
@click.option('--gae_lambda', required=False, type=float, default=0.95)
@click.option('--reward_norm_gamma', required=False, type=float, default=0.99)
@click.option('--reward_norm_cliprange', required=False, type=float, default=10.0)
@click.option('--coef_reward_completed', required=False, type=float, default=0.0)
@click.option('--coef_reward_assigned', required=False, type=float, default=1.0)
@click.option('--coef_reward_cancelled', required=False, type=float, default=0.0)
@click.option('--coef_reward_distance', required=False, type=float, default=1.0)
@click.option('--eval_num_simulator_steps', required=False, type=int, default=200)
@click.option('--eval_n_envs', required=False, type=int, default=2)
@click.option('--eval_trajectory_length', required=False, type=int, default=2000)
@click.option('--eval_epochs_frequency', required=False, type=int, default=1000)
@click.option('--eval_num_runs', required=False, type=int, default=5)
@click.option('--num_gambles_in_day', required=False, type=int, default=2880)
@click.option('--sweep_id', type=str, default='')
@click.option('--sweep_count', type=int, default=1)
def make_kwargs(**cfg):
    with open('configs/paths.json') as f:
        paths = dict(json.load(f))
        cfg.update(paths)

    train_id = str(uuid.uuid4().hex)
    cfg['train_id'] = train_id
    cfg['checkpoint_path'] += train_id + '.pt'
    cfg['history_db_path'] += train_id + '.db'
    cfg['debug_info_path'] += train_id + '.txt'
    return cfg


def run_ppo(**kwargs):
    if kwargs['mode'] == 'v1':
        maker = DeliveryMaker(**kwargs)
    elif kwargs['mode'] == 'v2':
        maker = DeliveryMaker2(**kwargs)
    else:
        raise RuntimeError(f"No such mode {kwargs['mode']}")
    if not kwargs['use_train_logs']:
        maker.ppo.metric_logger = None
    eval_runner = Runner(environment=maker.environment.copy(), actor_critic=maker.actor_critic,
                         n_envs=kwargs['eval_n_envs'], trajectory_length=kwargs['eval_trajectory_length'])
    inference_logger = InferenceMetricsRunner(runner=eval_runner, metric_logger=maker.metric_logger)
    dsp = NeuralSequantialDispatch(actor_critic=maker.actor_critic,
                                   max_num_points_in_route=kwargs['max_num_points_in_route'],
                                   use_dist=kwargs['use_dist'])

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
