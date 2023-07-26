from src.objects.point import Point
from src.networks.scoring_networks.net1 import ScoringNet
from src.networks.encoders.gamble_encoder import GambleTripleEncoder
from src.reinforcement.simulator_environment import SimulatorEnv
from src.reinforcement.custom_GAE import CustomGAE
from src.simulator.simulator import Simulator
from src.dispatch.dispatch import NeuralDispatch
from src.utils import get_batch_quality_metrics, get_CR
import json
import time
from tqdm import tqdm
import torch
from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule
from torch.distributions.categorical import Categorical
from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss


# if __name__ == "__main__":
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'device: {device}')

with open('configs/training_settings.json') as f:
    training_settings = json.load(f)
with open('configs/network_hyperparams.json') as f:
    hyperparams = json.load(f)
with open('configs/rl_settings.json') as f:
    rl_settings = json.load(f)

net = ScoringNet(
    n_layers=hyperparams['n_layers'],
    d_model=hyperparams['d_model'],
    n_head=hyperparams['n_head'],
    dim_ff=hyperparams['dim_ff'],
    device=device,
    path_weights='pretrained_models/eta_scoring_1/eta_scoring_1.pt'
)

encoder = GambleTripleEncoder(
    number_enc_dim=hyperparams['number_enc_dim'],
    d_model=hyperparams['d_model'],
    point_enc_dim=hyperparams['point_enc_dim'],
    path_weights='pretrained_models/assignment_cloning_model_v2/encoders/',
    # point_encoder=point_encoder,
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

optim = torch.optim.Adam(list(loss_module.parameters()) + list(encoder.parameters()), lr=rl_settings['lr'])
pbar = tqdm(total=rl_settings['total_frames'])

start_outer = time.time()
for i, tensordict_data in enumerate(collector):
    start_inner = time.time()
    print(f'outer time: {start_inner - start_outer}', end='')
    for _ in range(rl_settings['num_epochs']):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(rl_settings['frames_per_epoch'] // rl_settings['batch_size']):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), rl_settings['max_grad_norm'])
            optim.step()
            optim.zero_grad()
            # wandb.log({
            #     "loss_total": loss_value.item(),
            #     'loss_objective': loss_vals['loss_objective'],
            #     'loss_critic': loss_vals['loss_critic'],
            #     'loss_entropy': loss_vals['loss_entropy']
            #     })

    dsp = NeuralDispatch(net, encoder)
    cr = get_CR(get_batch_quality_metrics(dsp, Simulator,
                                            batch_size=training_settings['eval_batch_size'],
                                            num_steps=training_settings['eval_num_steps']))
    print('cr: ', cr)
    # wandb.log({'cr': cr})

    start_outer = time.time()
    print(f'inner time: {start_outer - start_inner}', end='\n')

# wandb.finish()
