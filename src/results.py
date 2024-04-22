import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch import nn
from scipy import stats

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
        eval_num_runs: int,
        reduce='mean',
        **kwargs
        ):
    print(f'Start {name}, {eval_num_runs} runs')
    db = Database(Path('history.db'))
    db.clear()
    results = evaluate(
        dispatch=dsp,
        run_id=0,
        reduce=reduce,
        eval_num_runs=eval_num_runs,
        simulator_cfg_path=kwargs['simulator_cfg_path'],
        sampler_mode=sample_mode,
        max_num_points_in_route=max_num_points_in_route,
        history_db_path='history.db',
        eval_num_simulator_steps=eval_num_simulator_steps,
        gif_path=kwargs['visualizations_path'] + name + '.gif',
        visualize=kwargs['visualize'],
        visualization_frequency=kwargs['visualization_frequency'],
        visualization_cgf_path=kwargs['visualization_cgf_path']
    )
    return results


def run_baselines(**kwargs) -> dict:
    baseline_dispatches = {
        'Hungarian': HungarianDispatch(DistanceScorer()),
        'Greedy': GreedyDispatch(DistanceScorer()),
        'Random': RandomDispatch(),
    }
    result_dict = {}
    kwargs['visualize'] = False
    for name, dsp in baseline_dispatches.items():
        result_dict[name] = make_runs(name, dsp, **kwargs)
    return result_dict


def run_model(checkpoint_id: str, **kwargs) -> dict:
    device = kwargs['device']
    model_size = kwargs['model_size']
    with open(kwargs['network_cfg_path']) as f:
        net_cfg = json.load(f)['encoder'][model_size]
    encoder = GambleEncoder(**net_cfg, **kwargs)
    ac = DeliveryActorCritic(gamble_encoder=encoder,
                             clm_emb_size=net_cfg['claim_embedding_dim'],
                             co_emb_size=net_cfg['courier_order_embedding_dim'],
                             gmb_emb_size=net_cfg['gamble_features_embedding_dim'],
                             device=device,
                             temperature=1.0,
                             use_dist=kwargs['use_dist'],
                             use_masks=kwargs['use_masks'])
    ac.load_state_dict(torch.load(kwargs['checkpoint_path'] + checkpoint_id + '.pt', map_location=device))
    dsp = NeuralSequantialDispatch(actor_critic=ac, max_num_points_in_route=kwargs['max_num_points_in_route'],
                                   use_dist=kwargs['use_dist'])
    return make_runs(checkpoint_id, dsp, **kwargs)


def compute_significancy(baseline_runs: dict[str, dict[str, list[float]]], model_runs: dict[str, list[float]],
                         metric: str, better_more: bool = True) -> str:
    best_baseline, best_value = '', -1e9
    for baseline_name, baseline_results in baseline_runs.items():
        value = float(np.mean(baseline_results[metric]))
        if (better_more and value > best_value) or (not better_more and value < best_value):
            best_value = value
            best_baseline = baseline_name

    baseline_values = baseline_runs[best_baseline][metric]
    model_values = model_runs[metric]
    pv = stats.ttest_ind(baseline_values, model_values, equal_var=False, nan_policy='raise', alternative='less').pvalue
    print('p-value:', pv)
    if pv < 0.01:
        return '***'
    elif pv < 0.05:
        return '**'
    elif pv < 0.1:
        return '*'
    return ''


def reduce_metrics(data: dict[str, list[float]], reduce='mean') -> dict[str, float]:
    result = {}
    for key, values in data.items():
        if reduce == 'mean':
            result[key] = mean(values)
    return result


def model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement()

    size_all_mb = (param_size + buffer_size) / 1000**2
    print('model size: {:.3f}M params'.format(size_all_mb))


def mean(values: list[float | None]) -> float | None:
    summ = 0.0
    for value in values:
        if value is None:
            return None
        summ += value
    return summ / len(values)