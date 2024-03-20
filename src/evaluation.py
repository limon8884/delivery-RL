from pathlib import Path

from src.simulator.simulator import Simulator
from src.simulator.data_reader import DataReader
from src.router_makers import AppendRouteMaker
from src.database.database import Database, Metric, DatabaseLogger
from src.dispatchs.base_dispatch import BaseDispatch


def evaluate(
    dispatch: BaseDispatch,
    run_id: int,
    **kwargs
) -> dict[Metric, float]:
    db_logger = DatabaseLogger(run_id=run_id)
    reader = DataReader.from_config(config_path=Path(kwargs['simulator_cfg_path']),
                                    sampler_mode=kwargs['sampler_mode'], db_logger=db_logger)
    route_maker = AppendRouteMaker(max_points_lenght=kwargs['max_num_points_in_route'], cutoff_radius=0.0)
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=Path(kwargs['simulator_cfg_path']),
                    db_logger=db_logger)
    sim.run(dispatch, num_iters=kwargs['eval_num_simulator_steps'])

    db = Database(Path(kwargs['history_db_path']))
    db.export_from_logger(db_logger)
    cr, ctd = db.get_metric(Metric.CR, run_id), db.get_metric(Metric.CTD, run_id)
    return {
        'CR': cr,
        'CTD': ctd,
    }
