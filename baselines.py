import click
import json
import pandas as pd
from src.results import BASELINES, BASELINE_NUM_RUNS, DEFAULT_RUN_ID
from src.results import make_evatuation_runs, evaluate_by_history, make_hist_path, mean


def run_baselines(**kwargs) -> None:
    for name, dsp in BASELINES.items():
        print(f'Baseline {name}')
        res = make_evatuation_runs(
            name,
            dsp,
            **kwargs
        )
        print(f'Results: {res}')


def eval_baselines(**kwargs):
    results = {}
    for baseline in BASELINES:
        baseline_hist_path = make_hist_path(baseline, history_db_path=kwargs['history_db_path'],
                                            sampler_mode=kwargs['sampler_mode'], eval_num_runs=BASELINE_NUM_RUNS,
                                            eval_num_simulator_steps=kwargs['eval_num_simulator_steps'])
        baseline_results = evaluate_by_history(run_id=DEFAULT_RUN_ID,
                                               history_db_path=baseline_hist_path,
                                               eval_num_runs=BASELINE_NUM_RUNS)
        results[baseline] = {k: mean(v) for k, v in baseline_results.items()}
    return results


@click.command()
@click.option('-s', 'sampler_mode', type=str, default="dummy_sampler")
@click.option('--n_pts', 'max_num_points_in_route', type=int, default=2)
@click.option('--steps', 'eval_num_simulator_steps', type=int, default=500)
# @click.option('--n_runs', 'eval_num_runs', type=int, default=5)
@click.option('--vis_freq', 'visualization_frequency', type=int, default=10)
@click.option('-v', '--visualize', type=bool, is_flag=True, default=False)
@click.option('-e', '--eval_only', type=bool, is_flag=True, default=False)
def main(**kwargs):
    kwargs['eval_num_runs'] = BASELINE_NUM_RUNS
    with open('configs/paths.json') as f:
        kwargs.update(json.load(f))
    if not kwargs['eval_only']:
        run_baselines(**kwargs)
    results = eval_baselines(**kwargs)
    print('-' * 50)
    print('Results', end='\n\n')
    df = pd.DataFrame.from_dict(results).T
    pd.set_option("display.precision", 3)
    print(df)


if __name__ == '__main__':
    main()
