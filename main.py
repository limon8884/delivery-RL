import json
from pathlib import Path

from src_new.simulator.simulator import Simulator
from src_new.simulator.data_reader import DataReader
from src_new.database.database import Database, Metric
from src_new.dispatchs.hungarian_dispatch import HungarianDispatch
from src_new.dispatchs.greedy_dispatch import GreedyDispatch
from src_new.dispatchs.scorers import DistanceScorer


def main():
    config_path = Path('configs_new/simulator.json')

    hungarian_db_path = Path('history_hung.db')
    greedy_db_path = Path('history_grd.db')

    db_h = Database(hungarian_db_path)
    db_g = Database(greedy_db_path)

    reader_h = DataReader.from_config(config_path=config_path, db=db_h)
    reader_g = DataReader.from_config(config_path=config_path, db=db_g)

    sim_h = Simulator(data_reader=reader_h, config_path=config_path, db=db_h)
    sim_g = Simulator(data_reader=reader_g, config_path=config_path, db=db_g)

    dsp_h = HungarianDispatch(DistanceScorer())
    dsp_g = GreedyDispatch(DistanceScorer())

    sim_h.run(dsp_h, num_iters=288)
    sim_g.run(dsp_g, num_iters=288)

    cr_h, ctd_h = db_h.get_metric(Metric.CR), db_h.get_metric(Metric.CTD)
    cr_g, ctd_g = db_g.get_metric(Metric.CR), db_g.get_metric(Metric.CTD)

    print(f'Hungarian: CR: {cr_h}, CTD: {ctd_h}')
    print(f'Greedy: CR: {cr_g}, CTD: {ctd_g}')


def main2():
    config_path = Path('configs_new/simulator.json')
    reader = DataReader.from_config(config_path=config_path, db=None)
    sim = Simulator(data_reader=reader, config_path=config_path, db=None)
    dsp = HungarianDispatch(DistanceScorer())
    sim.run(dsp, num_iters=288)


if __name__ == '__main__':
    main2()
