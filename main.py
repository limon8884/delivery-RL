import json
import click
import pandas as pd
from pprint import pprint

from src.results import run_baselines, run_model, compute_significancy, reduce_metrics


@click.command()
@click.option('--s_mode', 'sample_mode', type=str, default="dummy_sampler")
@click.option('--n_pts', 'max_num_points_in_route', type=int, default=2)
@click.option('--steps', 'eval_num_simulator_steps', type=int, default=200)
@click.option('--n_runs', 'eval_num_runs', type=int, default=5)
@click.option('--model_size', type=str, default='medium')
@click.option('--use_masks', type=bool, default=False)
@click.option('--use_dist', type=bool, default=True)
@click.option('--device', type=str, default='cpu')
@click.option('--use_pretrained_encoders', type=bool, default=True)
@click.option('--vis_freq', 'visualization_frequency', type=int, default=10)
@click.option('-v', '--visualize', type=bool, is_flag=True, default=False)
@click.argument('checkpoint_ids', nargs=-1)
def results(checkpoint_ids, **kwargs):
    print(f'Evaluating checkpoints: {checkpoint_ids} with kwargs:\n')
    pprint(kwargs)
    print('-' * 50 + '\n\n')
    print('Running baselines')
    with open('configs/paths.json') as f:
        kwargs.update(json.load(f))
    baseline_runs = run_baselines(reduce=None, **kwargs)
    results = {k: reduce_metrics(v) for k, v in baseline_runs.items()}
    for checkpoint_id in checkpoint_ids:
        print(f'Start running checkpoint {checkpoint_id}')
        model_runs = run_model(checkpoint_id, reduce=None, **kwargs)
        results[checkpoint_id] = reduce_metrics(model_runs)
        results[checkpoint_id]['significancy'] = compute_significancy(baseline_runs, model_runs,
                                                                      metric='CR', better_more=True)

    print('-' * 50)
    print('Results', end='\n\n')
    df = pd.DataFrame.from_dict(results).T
    pd.set_option("display.precision", 2)
    print(df)


if __name__ == '__main__':
    results()
