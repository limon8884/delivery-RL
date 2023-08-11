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

from torchrl.envs import Compose, TransformedEnv, StepCounter
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

my_env = TransformedEnv(
    my_env,
    Compose(
        StepCounter(),
    ),
)
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
    frames_per_batch=2,
    total_frames=8,
    split_trajs=False,
    # max_frames_per_traj=3,
    reset_at_each_iter=True,
    device=device,
)

for tensordict_data in tqdm(collector):
    print('step', tensordict_data['step_count'])
    print('next step', tensordict_data['next', 'step_count'])
    # print(tensordict_data['observation', 'ids', 'o'])
    # print(tensordict_data['observation', 'ids', 'c'])
    # print(tensordict_data['observation', 'ids', 'ar'])
    print('-' * 50)
