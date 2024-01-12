import json
from pathlib import Path

from src_new.simulator.simulator import Simulator
from src_new.simulator.data_reader import DataReader
from src_new.router_makers import AppendRouteMaker
from src_new.database.database import Database, Metric, Logger
from src_new.dispatchs.hungarian_dispatch import HungarianDispatch, BaseDispatch
from src_new.dispatchs.greedy_dispatch import GreedyDispatch
from src_new.dispatchs.scorers import DistanceScorer
from src_new.dispatchs.neural_sequantial_dispatch import NeuralSequantialDispatch
from src_new.networks.encoders import GambleEncoder
from src_new.networks.bodies import SimpleSequentialMLP


def run_dsp(dsp: BaseDispatch, config_path: Path, db_path: Path, run_id: int, max_num_points_in_route: int) -> None:
    logger = Logger(run_id=run_id)
    reader = DataReader.from_config(config_path=config_path, sampler_mode='dummy_sampler', logger=logger)
    route_maker = AppendRouteMaker(max_points_lenght=max_num_points_in_route, cutoff_radius=0.0)
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, logger=logger)
    try:
        sim.run(dsp, num_iters=288)
    finally:
        db = Database(db_path)
        db.export_from_logger(logger)
        # cr, ctd = db.get_metric(Metric.CR, run_id), db.get_metric(Metric.CTD, run_id)
        print(f'CR: {db.get_metric(Metric.CR, run_id)}\n \
            CTD: {db.get_metric(Metric.CTD, run_id)}\n \
            \n')


def main():
    simulator_config_path = Path('configs_new/simulator.json')
    network_config_path = Path('configs_new/network.json')
    db_path = Path('history.db')
    db = Database(db_path)
    db.clear()

    print('Hungarian')
    run_dsp(HungarianDispatch(DistanceScorer()), simulator_config_path, db_path, run_id=0, max_num_points_in_route=0)

    print('Greedy')
    run_dsp(GreedyDispatch(DistanceScorer()), simulator_config_path, db_path, run_id=1, max_num_points_in_route=0)

    print('Neural')
    max_num_points_in_route = 8
    with open(network_config_path) as f:
        net_cfg = json.load(f)['simple']
    encoder = GambleEncoder(
        courier_order_embedding_dim=net_cfg['courier_order_embedding_dim'],
        claim_embedding_dim=net_cfg['claim_embedding_dim'],
        courier_embedding_dim=net_cfg['courier_embedding_dim'],
        route_embedding_dim=net_cfg['route_embedding_dim'],
        point_embedding_dim=net_cfg['point_embedding_dim'],
        number_embedding_dim=net_cfg['number_embedding_dim'],
        max_num_points_in_route=max_num_points_in_route,
        device=None,
    )
    net = SimpleSequentialMLP(
        claim_embedding_dim=net_cfg['claim_embedding_dim'],
        courier_order_embedding_dim=net_cfg['courier_order_embedding_dim'],
        device=None,
    )
    run_dsp(NeuralSequantialDispatch(encoder=encoder, network=net, max_num_points_in_route=max_num_points_in_route),
            simulator_config_path, db_path, run_id=3,
            max_num_points_in_route=max_num_points_in_route)


def debug():
    db_path = Path('history.db')
    db = Database(db_path)
    res = db.select('select * from claims where claim_id = 191;')
    print(res)


if __name__ == '__main__':
    main()
    # debug()
