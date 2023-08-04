from src.dispatch.dispatch import NeuralDispatch
from src.simulator.simulator import Simulator
from src.objects.point import Point
from src.objects.gamble_triple import random_triple
from src.networks.scoring_networks.net1 import ScoringNet
from src.networks.encoders.gamble_encoder import GambleTripleEncoder
from src.networks.utils import (
    get_target_assignments,
    get_batch_embeddings_tensors,
    get_batch_masks,
    cross_entropy_assignment_loss,
    get_cross_mask,
    get_assignments_by_scores,
    compute_grad_norm
)
from src.utils import (
    get_batch_quality_metrics,
    get_CR,
    update_assignment_accuracy_statistics,
    update_run_counters,
    aggregate_metrics
)
from src.helpers.TimeLogger import TimeLogger

import torch
import json
import wandb
import numpy as np
from typing import List
from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed


# if __name__ == "__main__":
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'device: {device}')
update_run_counters(mode='cloning')

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
    device=device,
    path_weights=paths['pretrained_net']
)

encoder = GambleTripleEncoder(
    number_enc_dim=hyperparams['number_enc_dim'],
    d_model=hyperparams['d_model'],
    point_enc_dim=hyperparams['point_enc_dim'],
    path_weights=paths['pretrained_encoder'],
    device=device
)

bounds = (Point(0, 0), Point(10, 10))

# example
dsp = NeuralDispatch(net, encoder)
get_CR(get_batch_quality_metrics(dsp, Simulator, 1, 100))

wandb.login()
wandb.init(
    project="delivery-RL",
    name=f"training_cloning_{run_id}",
    config={
        'hyperparams': hyperparams,
        'training_settings': training_settings,
        'paths': paths,
        'device': device,
        })

time_logger = TimeLogger()

num_epochs = training_settings['num_epochs']
num_iters = training_settings['num_iters_in_epoch']
batch_size = training_settings['batch_size']
use_simulators = training_settings['use_simulators_instead_of_random_triples']
use_parallel = training_settings['use_parallel']

optimized_parameters = list(net.parameters()) + list(encoder.parameters())
if training_settings['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(optimized_parameters, lr=training_settings['lr'])
elif training_settings['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(optimized_parameters, lr=training_settings['lr'],
                                momentum=training_settings['momentum'])
else:
    raise RuntimeError('Unknown optimizer')

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=n_epochs, steps_per_epoch=n_iters)
scheduler = None


def apply_next_to_simulator(idx: int, simulators: List[Simulator], assignments):
    simulators[idx].Next(assignments[idx])


def apply_get_state_to_simulator(idx: int, simulators: List[Simulator]):
    return simulators[idx].GetState()


wandb_steps = {
    'train': 0,
    'simulator': 0,
    'eval': 0
}

for epoch in tqdm(range(num_epochs)):
    net.train()
    encoder.train()
    assignment_statistics = Counter()
    if use_simulators:
        simulators = [Simulator() for i in range(batch_size)]

    time_logger()
    for iter in range(num_iters):
        time_logger('loop')
        # generate training data
        if use_simulators:
            if use_parallel:
                triples = Parallel(n_jobs=-1)(delayed(apply_get_state_to_simulator)(i, simulators)
                                              for i in range(batch_size))
            else:
                triples = [sim.GetState() for sim in simulators]
        else:
            if use_parallel:
                triples = Parallel(n_jobs=-1)(delayed(random_triple)(bounds,
                                                                     max_items=training_settings['max_items_in_triple'])
                                              for i in range(batch_size))
            else:
                triples = [random_triple(bounds, max_items=training_settings['max_items_in_triple'])
                           for _ in range(batch_size)]
        max_num_ords = max([len(triple.orders) for triple in triples])
        max_num_crrs = max([len(triple.couriers) for triple in triples])
        time_logger('generating train data')

        # encode data
        target_assignment_idxs = []
        embeds = []
        ids = []
        for triple in triples:
            target_assignment_idxs.append(get_target_assignments(triple, max_num_ords, max_num_crrs))
            current_embeds, current_ids = encoder(triple, 0)
            embeds.append(current_embeds)
            ids.append(current_ids)
        batch_embs = get_batch_embeddings_tensors(embeds)
        batch_masks = get_batch_masks(triples, device=device)
        time_logger('encode data')

        # network forward pass
        optimizer.zero_grad()
        pred_scores, _ = net(batch_embs, batch_masks)
        time_logger('net pass')

        # gradient step
        loss = cross_entropy_assignment_loss(pred_scores, target_assignment_idxs, get_cross_mask(batch_masks))
        loss.backward()
        optimizer.step()
        time_logger('gradient step')

        # interraction with simulators
        if use_simulators:
            assignments_batch = get_assignments_by_scores(pred_scores, batch_masks, ids)
            if use_parallel:
                Parallel(n_jobs=-1)(delayed(apply_next_to_simulator)(i, simulators,
                                                                     assignments_batch) for i in range(batch_size))
            else:
                for i in range(batch_size):
                    simulators[i].Next(assignments_batch[i])
        time_logger('interraction with simulators')

        # get accuracy statistics
        for batch_idx in range(batch_size):
            update_assignment_accuracy_statistics(target_assignment_idxs[batch_idx],
                                                  pred_scores[batch_idx], assignment_statistics)
        time_logger('get accuracy statistics')

        # wandb update
        current_step = wandb.run.step
        wandb.log({
            'loss': loss.item(),
            'net_grad_norm': compute_grad_norm(net),
            'encoder_grad_norm': compute_grad_norm(encoder),
            'iter': wandb_steps['train']
        })
        wandb_steps['train'] += 1
        time_logger('send wandb statistics')

    # evaluation
    dsp = NeuralDispatch(net, encoder)
    simulator_metrics = get_batch_quality_metrics(dsp, Simulator,
                                                  batch_size=training_settings['eval_batch_size'],
                                                  num_steps=training_settings['eval_num_steps'])
    cr = get_CR(simulator_metrics)
    timings = time_logger.get_timings()

    wandb.log({'cr': cr, **assignment_statistics, **timings, 'iter': wandb_steps['eval']})
    wandb_steps['eval'] += 1

    for batch_metric in zip(*simulator_metrics):
        wandb.log({**aggregate_metrics(batch_metric, np.mean), 'iter:': wandb_steps['simulator']})
        wandb_steps['simulator'] += 1

    torch.save(net.state_dict(), paths['temporary'] + 'net.pt')
    torch.save(encoder.state_dict(), paths['temporary'] + 'encoder.pt')

wandb.finish()
