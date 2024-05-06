import json
import torch
import typing
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch import nn
from scipy import stats

from src.database.database import Database, Metric, DatabaseLogger
from src.dispatchs.hungarian_dispatch import HungarianDispatch, BaseDispatch
from src.dispatchs.greedy_dispatch import GreedyDispatch, GreedyDispatch2
from src.dispatchs.random_dispatch import RandomDispatch
from src.dispatchs.scorers import DistanceScorer
from src.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src.networks.encoders import GambleEncoder
from src.networks.claim_attention import ClaimAttention
from src.reinforcement.delivery import DeliveryActorCritic
from src.evaluation import evaluate, evaluate_by_history


BASELINES = {
    'Hungarian': HungarianDispatch(DistanceScorer()),
    'Greedy': GreedyDispatch(DistanceScorer()),
    'Greedy2': GreedyDispatch2(DistanceScorer()),
    'Random': RandomDispatch(),
}
DEFAULT_RUN_ID = 0
BASELINE_NUM_RUNS = 10
# SIGNIFICANCY_METRIC = 'CR'


def make_evatuation_runs(
        model_id: str,
        dsp: BaseDispatch,
        **kwargs
        ) -> dict[str, typing.Optional[float]]:
    '''
    history_db_path - path to folder, not file
    '''
    print(f'Start {model_id}, {kwargs["eval_num_runs"]} runs')
    db_path = make_hist_path(model_id, **kwargs)
    kwargs['history_db_path'] = db_path
    db = Database(db_path)
    db.clear()
    results = evaluate(
        dispatch=dsp,
        run_id=DEFAULT_RUN_ID,
        gif_path=make_gif_path(model_id, **kwargs),
        **kwargs
    )
    print('Results:', results)
    return results


def run_model(checkpoint_id: str, **kwargs) -> None:
    device = kwargs['device']
    model_size = kwargs['model_size']
    with open(kwargs['network_cfg_path']) as f:
        net_cfg = json.load(f)
        encoder_cfg = net_cfg['encoder'][model_size]
        attn_cfg = net_cfg['attention'][model_size]
    encoder = GambleEncoder(**encoder_cfg, **kwargs)
    attention = nn.Transformer(batch_first=True, **kwargs, **attn_cfg).to(device) if kwargs['use_attn'] else None
    ac = DeliveryActorCritic(gamble_encoder=encoder, attention=attention,
                             clm_emb_size=encoder_cfg['claim_embedding_dim'],
                             co_emb_size=encoder_cfg['courier_order_embedding_dim'],
                             gmb_emb_size=encoder_cfg['gamble_features_embedding_dim'],
                             exploration_temperature=1.0,
                             **kwargs)
    ac.load_state_dict(torch.load(kwargs['checkpoint_path'] + checkpoint_id + '.pt', map_location=device))
    dsp = NeuralSequantialDispatch(actor_critic=ac, **kwargs)
    make_evatuation_runs(checkpoint_id, dsp, **kwargs)


def eval_model(model_id: str, **kwargs) -> dict[str, dict[str, typing.Any]]:
    results: dict[str, dict[str, typing.Any]] = {}
    model_hist_path = make_hist_path(model_id, **kwargs)
    if not model_hist_path.is_file():
        run_model(model_id, **kwargs)
    model_results = evaluate_by_history(run_id=DEFAULT_RUN_ID, eval_num_runs=kwargs['eval_num_runs'],
                                        history_db_path=model_hist_path)
    results[model_id] = {k: mean(v) for k, v in model_results.items()}
    for baseline in BASELINES:
        baseline_hist_path = make_hist_path(baseline, history_db_path=kwargs['history_db_path'],
                                            sampler_mode=kwargs['sampler_mode'], eval_num_runs=BASELINE_NUM_RUNS,
                                            eval_num_simulator_steps=kwargs['eval_num_simulator_steps'])
        baseline_results = evaluate_by_history(run_id=DEFAULT_RUN_ID,
                                               history_db_path=baseline_hist_path,
                                               eval_num_runs=BASELINE_NUM_RUNS)
        # print(baseline, baseline_results)
        results[baseline] = compute_significancy(baseline_results, model_results)
    return results


def compute_significancy(baseline_runs: dict[str, list[typing.Optional[float]]],
                         model_runs: dict[str, list[typing.Optional[float]]]) -> dict[str, str]:
    results: dict[str, str] = {}
    for metric in model_runs:
        baseline_values = baseline_runs[metric]
        model_values = model_runs[metric]
        mean_value = mean(baseline_values)
        if mean_value is None:
            results[metric] = 'NAN'
            continue
        pvalue = stats.ttest_ind(baseline_values, model_values, equal_var=False, nan_policy='raise',
                                 alternative='two-sided').pvalue
        results[metric] = represent_significancy(mean_value, pvalue)
    return results


def represent_significancy(value: float, pvalue: float) -> str:
    if pvalue < 0.01:
        return f'{value:.3f}***'
    elif pvalue < 0.05:
        return f'{value:.3f} **'
    elif pvalue < 0.1:
        return f'{value:.3f}  *'
    return f'{value:.3f}   '


def make_hist_path(model_id: str, history_db_path: str, sampler_mode: str, eval_num_runs: int,
                   eval_num_simulator_steps: int, **kwargs) -> Path:
    file_name = '_'.join([model_id, str(eval_num_simulator_steps), str(eval_num_runs)]) + '.db'
    return Path(history_db_path + sampler_mode + '/' + file_name)


def make_gif_path(model_id: str, visualizations_path: str, sampler_mode: str, **kwargs) -> Path:
    return Path(visualizations_path + sampler_mode + '/' + model_id + '.gif')


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
