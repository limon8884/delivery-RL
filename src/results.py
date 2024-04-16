import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch import nn

from src.database.database import Database, Metric, DatabaseLogger
from src.dispatchs.hungarian_dispatch import HungarianDispatch, BaseDispatch
from src.dispatchs.greedy_dispatch import GreedyDispatch
from src.dispatchs.random_dispatch import RandomDispatch
from src.dispatchs.scorers import DistanceScorer
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.networks.encoders import GambleEncoder
from src.reinforcement.delivery import DeliveryActorCritic
from src.evaluation import evaluate


def make_runs(
        name: str,
        dsp: BaseDispatch,
        sample_mode: str,
        max_num_points_in_route: int,
        eval_num_simulator_steps: int,
        n_runs: int,
        **kwargs
        ):
    print(f'Start {name}, {n_runs} runs')
    db = Database(Path('history.db'))
    db.clear()
    results = defaultdict(list)
    for run_id in range(n_runs):
        res = evaluate(
            dispatch=dsp,
            run_id=run_id,
            simulator_cfg_path=kwargs['simulator_cfg_path'],
            sampler_mode=sample_mode,
            max_num_points_in_route=max_num_points_in_route,
            history_db_path='history.db',
            eval_num_simulator_steps=eval_num_simulator_steps,
        )
        for k, v in res.items():
            results[k].append(v)
    return {k: np.mean(v_list) for k, v_list in results.items()}


def run_baselines(**kwargs) -> dict:
    baseline_dispatches = {
        'Hungarian': HungarianDispatch(DistanceScorer()),
        'Greedy': GreedyDispatch(DistanceScorer()),
        'Random': RandomDispatch(),
    }
    result_dict = {}
    for name, dsp in baseline_dispatches.items():
        result_dict[name] = make_runs(name, dsp, **kwargs)
    return result_dict


def run_model(checkpoint_id: str, **kwargs) -> dict:
    device = kwargs['device']
    model_size = kwargs['model_size']
    use_dist_feature = kwargs['use_dist_feature']
    with open(kwargs['network_cfg_path']) as f:
        net_cfg = json.load(f)['encoder'][model_size]
    encoder = GambleEncoder(**net_cfg, **kwargs)
    coc_emb_size = net_cfg['claim_embedding_dim'] + net_cfg['courier_order_embedding_dim']
    ac = DeliveryActorCritic(gamble_encoder=encoder, coc_emb_size=coc_emb_size, device=device,
                             temperature=1.0, use_dist_feature=use_dist_feature)
    ac.load_state_dict(torch.load(kwargs['checkpoint_path'] + checkpoint_id + '.pt', map_location=device))
    dsp = NeuralSequantialDispatch(actor_critic=ac, max_num_points_in_route=kwargs['max_num_points_in_route'])
    return make_runs(checkpoint_id, dsp, **kwargs)


def model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement()

    size_all_mb = (param_size + buffer_size) / 1000**2
    print('model size: {:.3f}M params'.format(size_all_mb))


def results():
    cfg = {
        "sample_mode": 'dummy_sampler',
        "max_num_points_in_route": 2,
        "eval_num_simulator_steps": 200,
        "n_runs": 1
    }
    df = pd.DataFrame.from_dict(run_baselines(**cfg)).T
    pd.set_option("display.precision", 2)
    print(df)
