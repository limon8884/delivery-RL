from src.objects.point import Point
from src.networks.scoring_networks.net1 import ScoringNet
from src.networks.encoders.gamble_encoder import GambleTripleEncoder
from src.reinforcement.simulator_environment import SimulatorEnv
from src.reinforcement.custom_GAE import CustomGAE
from src.simulator.simulator import Simulator
from src.dispatch.dispatch import NeuralDispatch
from src.networks.utils import (
    compute_grad_norm
)
from src.utils import (
    get_batch_quality_metrics,
    get_CR,
    update_run_counters,
    aggregate_metrics, 
    repr_big_number
)
from src.helpers.TimeLogger import TimeLogger

from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule
from torch.distributions.categorical import Categorical
from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torch.optim.lr_scheduler import OneCycleLR

import torch
import json
import wandb
import numpy as np
from tqdm import tqdm


# if __name__ == "__main__":
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'device: {device}')
update_run_counters(mode='RL')

with open('configs/network_hyperparams.json') as f:
    hyperparams = json.load(f)
with open('configs/rl_settings.json') as f:
    rl_settings = json.load(f)
with open('configs/run_ids.json') as f:
    run_id = json.load(f)['RL']
with open('configs/paths.json') as f:
    paths = json.load(f)

net = ScoringNet(
    n_layers=hyperparams['n_layers'],
    d_model=hyperparams['d_model'],
    n_head=hyperparams['n_head'],
    dim_ff=hyperparams['dim_ff'],
    path_weights=paths['pretrained_net'] if rl_settings['use_pretrained_net'] else None,
    device=device
)

encoder = GambleTripleEncoder(
    number_enc_dim=hyperparams['number_enc_dim'],
    d_model=hyperparams['d_model'],
    point_enc_dim=hyperparams['point_enc_dim'],
    path_weights=paths['pretrained_encoder'] if rl_settings['use_pretrained_encoder'] else None,
    device=device
)
encoder.eval()

bounds = (Point(0, 0), Point(10, 10))

my_env = SimulatorEnv(Simulator, encoder)
check_env_specs(my_env)

module = TensorDictModule(
    net, in_keys=[('observation', 'tensors'), ('observation', 'masks')], out_keys=['logits', 'state_value']
)

policy_module_actor = ProbabilisticActor(
    module=module,
    in_keys=["logits"],
    distribution_class=Categorical,
    return_log_prob=True,
)

collector = SyncDataCollector(
    my_env,
    policy_module_actor,
    frames_per_batch=rl_settings['frames_per_epoch'],
    total_frames=rl_settings['total_frames'],
    split_trajs=False,
    max_frames_per_traj=rl_settings['max_trajectory_length'],
    reset_at_each_iter=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(rl_settings['frames_per_epoch']),
    sampler=SamplerWithoutReplacement(),
    batch_size=rl_settings['batch_size']
)

advantage_module = CustomGAE(
    gamma=rl_settings['gamma'],
    lmbda=rl_settings['lmbda'],
    value_network=module,
    average_gae=True,
    value_key='state_value'
)

loss_module = ClipPPOLoss(
    actor=policy_module_actor,
    critic=policy_module_actor,
    advantage_key="advantage",
    clip_epsilon=rl_settings['clip_epsilon'],
    entropy_bonus=bool(rl_settings['entropy_eps']),
    entropy_coef=rl_settings['entropy_eps'],
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
)

wandb.login()
wandb.init(
    project="delivery-RL",
    name=f"training_RL_{run_id}",
    config={
        'hyperparams': hyperparams,
        'rl_settings': rl_settings,
        'paths': paths,
        'device': device,
        })

time_logger = TimeLogger('Time: ')

num_epochs = rl_settings['num_epochs']
batch_size = rl_settings['batch_size']
total_frames = rl_settings['total_frames']
frames_per_epoch = rl_settings['frames_per_epoch']
total_iters = num_epochs * total_frames

optimized_parameters = list(loss_module.parameters())
if rl_settings['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(optimized_parameters, lr=rl_settings['lr'])
elif rl_settings['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(optimized_parameters, lr=rl_settings['lr'],
                                momentum=rl_settings['momentum'])
else:
    raise RuntimeError('Unknown optimizer')

if rl_settings['scheduler'] is None:
    scheduler = None
elif rl_settings['scheduler'] == 'OneCycle':
    scheduler = OneCycleLR(optimizer, max_lr=rl_settings['max_lr'], total_steps=total_iters)
else:
    raise RuntimeError('Unknown scheduler')

wandb_steps = {
    'train': 0,
    'simulator': 0,
    'eval': 0,
    'outer': 0
}

print('Starting training! Total iters: ' + repr_big_number(total_iters) +
      ', total epochs: ' + repr_big_number(total_frames // frames_per_epoch * num_epochs))
time_logger()
for collector_iter, tensordict_data in enumerate(collector):
    time_logger('collector')
    for epoch in range(num_epochs):
        net.train()
        with torch.no_grad():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        for iter in range(frames_per_epoch // batch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(optimized_parameters, rl_settings['max_grad_norm'])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            wandb.log({
                'net_grad_norm': compute_grad_norm(net),
                'encoder_grad_norm': compute_grad_norm(encoder),
                'lr': scheduler.get_lr() if scheduler is not None else rl_settings['lr'],
                "loss_total": loss_value.item(),
                'loss_objective': loss_vals['loss_objective'].item(),
                'loss_critic': loss_vals['loss_critic'].item(),
                'loss_entropy': loss_vals['loss_entropy'].item(),
                'iter': wandb_steps['train']
                })
            wandb_steps['train'] += 1
    time_logger('epochs')

    # evaluation
    if collector_iter % rl_settings['eval_freq'] == 0:
        dsp = NeuralDispatch(net, encoder)
        net.eval()
        simulator_metrics = get_batch_quality_metrics(dsp, Simulator,
                                                      batch_size=rl_settings['eval_batch_size'],
                                                      num_steps=rl_settings['eval_num_steps'])
        cr = get_CR(simulator_metrics)
        for batch_metric in zip(*simulator_metrics):
            wandb.log({**aggregate_metrics(batch_metric, np.mean), 'iter:': wandb_steps['simulator']})
            wandb_steps['simulator'] += 1
        time_logger('evaluation')

    torch.save(net.state_dict(), paths['temporary'] + 'net.pt')
    torch.save(encoder.state_dict(), paths['temporary'] + 'encoder.pt')
    time_logger('save models')

    wandb.log({'cr': cr, **time_logger.get_timings(), 'iter': wandb_steps['eval']})
    wandb_steps['eval'] += 1
wandb.finish()
