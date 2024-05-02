import json
import click
import pandas as pd
from pprint import pprint

from src.results import eval_model


@click.command()
@click.option('-s', 'sampler_mode', type=str, default="dummy_sampler")
@click.option('--n_pts', 'max_num_points_in_route', type=int, default=2)
@click.option('--steps', 'eval_num_simulator_steps', type=int, default=200)
@click.option('--n_runs', 'eval_num_runs', type=int, default=5)
@click.option('--model_size', type=str, default='medium')
@click.option('--use_masks', type=bool, default=False)
@click.option('--use_dist', type=bool, default=True)
@click.option('--mask_fake_crr', type=bool, default=False)
@click.option('--use_attn', type=bool, default=False)
@click.option('--use_route', type=bool, default=False)
@click.option('--device', type=str, default='cpu')
@click.option('--use_pretrained_encoders', type=bool, default=True)
@click.option('--normalize_coords', required=False, type=bool, default=False)
@click.option('--disable_features', type=bool, default=False)
@click.option('--vis_freq', 'visualization_frequency', type=int, default=10)
@click.option('-v', '--visualize', type=bool, is_flag=True, default=False)
@click.option('--distance_norm_constant', required=False, type=float, default=1.0)
@click.option('--time_norm_constant', required=False, type=float, default=600.0)
@click.option('--num_norm_constant', required=False, type=float, default=100.0)
@click.argument('checkpoint_id', nargs=1)
def results(checkpoint_id, **kwargs):
    with open('configs/paths.json') as f:
        kwargs.update(json.load(f))
    results = eval_model(checkpoint_id, **kwargs)
    print('-' * 50)
    print('Results', end='\n\n')
    df = pd.DataFrame.from_dict(results).T
    pd.set_option("display.precision", 3)
    print(df)


if __name__ == '__main__':
    results()
