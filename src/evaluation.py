# import numpy as np
import typing
from pathlib import Path

from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.router_makers import AppendRouteMaker
from src.database.database import Database, Metric, DatabaseLogger
from src.dispatchs.base_dispatch import BaseDispatch
from src.visualization import Visualization


MAX_NUM_EVAL_RUNS = 10


def evaluate(
    dispatch: BaseDispatch,
    run_id: int,
    eval_num_runs: int,
    visualize: bool = False,
    **kwargs
) -> dict[str, typing.Optional[float]]:
    assert eval_num_runs <= MAX_NUM_EVAL_RUNS
    results: dict[str, list[typing.Optional[float]]] = {
        'CR': [],
        'CTD': [],
        'arrival_dist': [],
    }
    for run_index in range(eval_num_runs):
        visualizer = Visualization(
            config_path=kwargs['visualization_cgf_path'],
            vis_freq=kwargs['visualization_frequency']
            ) if (visualize and run_index == 0) else None
        local_run_id = run_id * MAX_NUM_EVAL_RUNS + run_index
        db_logger = DatabaseLogger(run_id=local_run_id)
        reader = DataReader.from_config(config_path=Path(kwargs['simulator_cfg_path']),
                                        sampler_mode=kwargs['sampler_mode'], db_logger=db_logger)
        route_maker = AppendRouteMaker(max_points_lenght=kwargs['max_num_points_in_route'], cutoff_radius=0.0)
        sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=Path(kwargs['simulator_cfg_path']),
                        db_logger=db_logger)
        sim.run(dispatch, num_iters=kwargs['num_simulator_steps'], visualization=visualizer)

        db = Database(Path(kwargs['history_db_path']))
        db.export_from_logger(db_logger)
        results['CR'].append(db.get_metric(Metric.CR, local_run_id))
        results['CTD'].append(db.get_metric(Metric.CTD, local_run_id))
        results['arrival_dist'].append(db.get_metric(Metric.NOT_BATCHED_ARRIVAL_DISTANCE, local_run_id))
        if visualizer is not None:
            visualizer.to_gif(kwargs['gif_path'], duration_sec=1)

    return {k: mean(v) for k, v in results.items()}


def evaluate_by_history(
    run_id: int,
    eval_num_runs: int,
    history_db_path: Path,
) -> dict[str, list[typing.Optional[float]]]:
    assert eval_num_runs <= MAX_NUM_EVAL_RUNS
    results: dict[str, list[typing.Optional[float]]] = {
        'CR': [],
        'CTD': [],
        'arrival_dist': [],
    }
    if not history_db_path.exists():
        raise RuntimeError(f'Path {history_db_path} does not exist')
    db = Database(history_db_path)
    for run_index in range(eval_num_runs):
        local_run_id = run_id * MAX_NUM_EVAL_RUNS + run_index
        results['CR'].append(db.get_metric(Metric.CR, local_run_id))
        results['CTD'].append(db.get_metric(Metric.CTD, local_run_id))
        results['arrival_dist'].append(db.get_metric(Metric.NOT_BATCHED_ARRIVAL_DISTANCE, local_run_id))
    return results


def mean(values: list[typing.Optional[float]]) -> typing.Optional[float]:
    summ = 0.0
    for value in values:
        if value is None:
            return None
        summ += value
    return summ / len(values)
