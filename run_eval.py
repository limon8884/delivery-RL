from src.dispatch.dispatch import NeuralDispatch
from src.simulator.simulator import Simulator
from src.objects.point import Point
from src.networks.scoring_networks.net1 import ScoringNet
from src.networks.encoders.gamble_encoder import GambleTripleEncoder

from src.utils import (
    get_batch_quality_metrics,
    get_CR,
    aggregate_metrics
)


import torch
import json
import numpy as np


# if __name__ == "__main__":
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'device: {device}')

with open('configs/training_settings.json') as f:
    training_settings = json.load(f)
with open('configs/network_hyperparams.json') as f:
    hyperparams = json.load(f)
with open('configs/run_ids.json') as f:
    run_id = json.load(f)['cloning']
with open('configs/paths.json') as f:
    paths = json.load(f)

net = ScoringNet(
    n_layers=hyperparams['n_layers'],
    d_model=hyperparams['d_model'],
    n_head=hyperparams['n_head'],
    dim_ff=hyperparams['dim_ff'],
    path_weights=paths['pretrained_net'] if training_settings['use_pretrained'] else None,
    device=device
)

encoder = GambleTripleEncoder(
    number_enc_dim=hyperparams['number_enc_dim'],
    d_model=hyperparams['d_model'],
    point_enc_dim=hyperparams['point_enc_dim'],
    path_weights=paths['pretrained_encoder'] if training_settings['use_pretrained'] else None,
    device=device
)

bounds = (Point(0, 0), Point(10, 10))

# example
dsp = NeuralDispatch(net, encoder)
net.eval()
encoder.eval()
simulator_metrics = get_batch_quality_metrics(dsp, Simulator,
                                              batch_size=8,
                                              num_steps=100)
cr = get_CR(simulator_metrics)

for iter, batch_metric in enumerate(zip(*simulator_metrics)):
    print('iter: ' + str(iter))
    for k, v in aggregate_metrics(batch_metric, np.mean).items():
        print(k, v)
    print('-' * 50)

print(cr, end='\n\n')
