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
    aggregate_metrics
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
    path_weights=paths['pretrained_net'] if rl_settings['use_pretrained'] else None,
    device=device
)

encoder = GambleTripleEncoder(
    number_enc_dim=hyperparams['number_enc_dim'],
    d_model=hyperparams['d_model'],
    point_enc_dim=hyperparams['point_enc_dim'],
    path_weights=paths['pretrained_encoder'] if rl_settings['use_pretrained'] else None,
    device=device
)

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

inner_time_logger = TimeLogger()
outer_time_logger = TimeLogger()

num_epochs = rl_settings['num_epochs']
batch_size = rl_settings['batch_size']
total_frames = rl_settings['total_frames']
frames_per_epoch = rl_settings['frames_per_epoch']


optimized_parameters = list(loss_module.parameters()) + list(encoder.parameters())
if rl_settings['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(optimized_parameters, lr=rl_settings['lr'])
elif rl_settings['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(optimized_parameters, lr=rl_settings['lr'],
                                momentum=rl_settings['momentum'])
else:
    raise RuntimeError('Unknown optimizer')

wandb_steps = {
    'train': 0,
    'simulator': 0,
    'eval': 0,
    'outer': 0
}

outer_time_logger()
for tensordict_data in tqdm(collector):
    print('collected!')
    outer_time_logger('collector')
    inner_time_logger()
    for epoch in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        net.train()
        encoder.train()
        # assignment_statistics = Counter()
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())

        for iter in range(frames_per_epoch // batch_size):
            inner_time_logger('loop')
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            inner_time_logger('forward pass')

            loss_value.backward()
            # torch.nn.utils.clip_grad_norm_(optimized_parameters, rl_settings['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
            inner_time_logger('gradient step')

            wandb.log({
                "loss_total": loss_value.item(),
                'loss_objective': loss_vals['loss_objective'].item(),
                'loss_critic': loss_vals['loss_critic'].item(),
                'loss_entropy': loss_vals['loss_entropy'].item(),
                'net_grad_norm': compute_grad_norm(net),
                'encoder_grad_norm': compute_grad_norm(encoder),
                'iter': wandb_steps['train']
                })
            wandb_steps['train'] += 1
            inner_time_logger('send wandb statistics')

        # evaluation
        dsp = NeuralDispatch(net, encoder)
        simulator_metrics = get_batch_quality_metrics(dsp, Simulator,
                                                      batch_size=rl_settings['eval_batch_size'],
                                                      num_steps=rl_settings['eval_num_steps'])
        cr = get_CR(simulator_metrics)
        timings = inner_time_logger.get_timings()

        wandb.log({'cr': cr, **timings, 'iter': wandb_steps['eval']})
        wandb_steps['eval'] += 1

        for batch_metric in zip(*simulator_metrics):
            wandb.log({**aggregate_metrics(batch_metric, np.mean), 'iter:': wandb_steps['simulator']})
            wandb_steps['simulator'] += 1

        torch.save(net.state_dict(), paths['temporary'] + 'net.pt')
        torch.save(encoder.state_dict(), paths['temporary'] + 'encoder.pt')

    wandb.log({**outer_time_logger.get_timings(), 'iter': wandb_steps['outer']})
    wandb_steps['outer'] += 1
    outer_time_logger('epoch')
wandb.finish()
