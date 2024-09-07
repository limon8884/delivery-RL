import json
import click
import uuid
import torch
import numpy
import random
import wandb
from tqdm import tqdm

from src.reinforcement.base import Runner, InferenceMetricsRunner
from src.reinforcement.lunar_lander import GymMaker, GymEnv, GymActorCritic


@click.command()
@click.option('--group_run', '-g', 'group_run', type=str)
@click.option('--description', '-d', 'description', type=str)
@click.option('--total_iters', required=False, type=int)
@click.option('--device', required=False, type=str)
@click.option('--use_wandb', required=False, type=bool)
@click.option('--fix_zero_seed', required=False, type=bool)
@click.option('--n_envs', required=False, type=int)
@click.option('--trajectory_length', required=False, type=int)
@click.option('--batch_size', required=False, type=int)
@click.option('--num_epochs_per_traj', required=False, type=int)
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
@click.option('--num_simulator_steps', required=False, type=int)
@click.option('--eval_n_envs', required=False, type=int)
@click.option('--eval_trajectory_length', required=False, type=int)
@click.option('--eval_epochs_frequency', required=False, type=int)
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


def evaluate(
    actor_critic: GymActorCritic,
    **kwargs
) -> dict[str, float]:
    env = GymEnv(gym_name=kwargs['env_name'])
    state = env.reset()
    total_reward = 0
    step = 0
    while step < kwargs['num_simulator_steps']:
        actor_critic([state])
        action = actor_critic.get_actions_list(best_actions=True)[0]
        state, reward, reset, _ = env.step(action)
        total_reward += reward
        step += 1
        if reset:
            break

    return {
        'total_reward': total_reward,
        'num_steps': step,
    }


def run_ppo(**kwargs):
    kwargs['env_name'] = 'LunarLander-v2'
    maker = GymMaker(**kwargs)
    # maker.ppo.logger = None

    eval_runner = Runner(environment=maker.environment, actor_critic=maker.actor_critic,
                         n_envs=kwargs['eval_n_envs'], trajectory_length=kwargs['eval_trajectory_length'])
    inference_logger = InferenceMetricsRunner(runner=eval_runner, metric_logger=maker.metric_logger)

    for iteration in tqdm(range(kwargs['total_iters'])):
        maker.actor_critic.train()
        sample = maker.sampler.sample()
        maker.ppo.step(sample)
        if (iteration + 1) % kwargs['eval_epochs_frequency'] == 0:
            maker.actor_critic.eval()
            metrics = evaluate(actor_critic=maker.actor_critic, **kwargs)
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
            project="Lunar-lander",
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
