import json
from pathlib import Path

from src_new.simulator.simulator import Simulator
from src_new.simulator.data_reader import DataReader
from src_new.router_makers import BaseRouteMaker
from src_new.database.database import Database, Metric, Logger
from src_new.dispatchs.hungarian_dispatch import HungarianDispatch, BaseDispatch
from src_new.dispatchs.greedy_dispatch import GreedyDispatch
from src_new.dispatchs.scorers import DistanceScorer


def run_dsp(dsp: BaseDispatch, config_path: Path, db_path: Path, run_id: int) -> None:
    logger = Logger(run_id=run_id)
    reader = DataReader.from_config(config_path=config_path, logger=logger)
    route_maker = BaseRouteMaker(max_points_lenght=0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, logger=logger)
    sim.run(dsp, num_iters=288)
    db = Database(db_path)
    db.export_from_logger(logger)
    cr, ctd = db.get_metric(Metric.CR, run_id), db.get_metric(Metric.CTD, run_id)
    print(f'CR: {cr}, CTD: {ctd}')


def main():
    config_path = Path('configs_new/simulator.json')
    db_path = Path('history.db')
    db = Database(db_path)
    db.clear()

    print('Hungarian')
    run_dsp(HungarianDispatch(DistanceScorer()), config_path, db_path, run_id=0)
    print('Greedy')
    run_dsp(GreedyDispatch(DistanceScorer()), config_path, db_path, run_id=1)


if __name__ == '__main__':
    main()
