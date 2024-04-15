import json
import random
from datetime import datetime, timedelta
from pytest import approx
from pathlib import Path

from src.simulator.simulator import DataReader, Simulator
from src.router_makers import BaseRouteMaker
from src.dispatchs.hungarian_dispatch import HungarianDispatch
from src.dispatchs.scorers import DistanceScorer
from src.database.database import Database, Metric
from src.database.logger import DatabaseLogger


BASE_DTTM = datetime.utcnow()

TEST_DATA_COURIERS_CR = [
    {'start_dttm': BASE_DTTM + timedelta(seconds=0), 'end_dttm': BASE_DTTM + timedelta(seconds=1000),
     'start_position_lat': 0.0, 'start_position_lon': 0.0},
]

TEST_DATA_CLAIMS_CR = [
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=19),
        'source_point_lat': 0.0,
        'source_point_lon': 0.0,
        'destination_point_lat': 1.0,
        'destination_point_lon': 1.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=19),
        'source_point_lat': 1.0,
        'source_point_lon': 0.0,
        'destination_point_lat': 1.0,
        'destination_point_lon': 1.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=19),
        'source_point_lat': 0.0,
        'source_point_lon': 1.0,
        'destination_point_lat': 1.0,
        'destination_point_lon': 1.0,
        'waiting_on_point_source': timedelta(seconds=0),
        'waiting_on_point_destination': timedelta(seconds=0),
    },
]


def test_cr_simple(tmp_path):
    config_path = tmp_path / 'tmp.json'
    db_path = tmp_path / 'test_db.db'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 10,
            'courier_speed': 0.1
        }
        json.dump(config, f)

    db_logger = DatabaseLogger(run_id=-1)
    reader = DataReader.from_list(TEST_DATA_COURIERS_CR, TEST_DATA_CLAIMS_CR, db_logger=db_logger)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=db_logger)
    dsp = HungarianDispatch(DistanceScorer())

    sim.run(dsp, num_iters=5)

    db = Database(db_path)
    db.export_from_logger(db_logger)

    assert db.get_metric(Metric.CR, run_id=-1) == approx(1/3, 0.000001)


def test_cr_100_percent(tmp_path):
    n_iters = 100
    config_path = tmp_path / 'tmp.json'
    db_path = tmp_path / 'test_db.db'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 1,
            'courier_speed': 1.0
        }
        json.dump(config, f)

    couriers = [
        {
            'start_dttm': BASE_DTTM + timedelta(seconds=i),
            'end_dttm': BASE_DTTM + timedelta(seconds=i + 2),
            'start_position_lat': random.random(),
            'start_position_lon': random.random()
        }
        for i in range(n_iters)
    ]
    claims = [
        {
            'created_dttm': BASE_DTTM + timedelta(seconds=i),
            'cancelled_dttm': BASE_DTTM + timedelta(seconds=i + 2),
            'source_point_lat': random.random(),
            'source_point_lon': random.random(),
            'destination_point_lat': random.random(),
            'destination_point_lon': random.random(),
            'waiting_on_point_source': timedelta(seconds=0),
            'waiting_on_point_destination': timedelta(seconds=0),
        }
        for i in range(n_iters)
    ]

    db_logger = DatabaseLogger(run_id=-1)
    reader = DataReader.from_list(couriers, claims, db_logger=db_logger)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=db_logger)
    dsp = HungarianDispatch(DistanceScorer())

    sim.run(dsp, num_iters=n_iters + 5)

    db = Database(db_path)
    db.export_from_logger(db_logger)
    assert db.get_metric(Metric.CR, run_id=-1) == approx(1.0, 0.000001)


TEST_DATA_COURIERS_CTD = [
    {'start_dttm': BASE_DTTM + timedelta(seconds=0), 'end_dttm': BASE_DTTM + timedelta(seconds=100),
     'start_position_lat': 0.0, 'start_position_lon': 0.0},
    {'start_dttm': BASE_DTTM + timedelta(seconds=0), 'end_dttm': BASE_DTTM + timedelta(seconds=100),
     'start_position_lat': 0.5, 'start_position_lon': 0.0},
    {'start_dttm': BASE_DTTM + timedelta(seconds=0), 'end_dttm': BASE_DTTM + timedelta(seconds=100),
     'start_position_lat': 1.0, 'start_position_lon': 0.0},
]

TEST_DATA_CLAIMS_CTD = [
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=100),
        'source_point_lat': 0.0,
        'source_point_lon': 0.0,
        'destination_point_lat': 0.0,
        'destination_point_lon': 0.0,
        'waiting_on_point_source': timedelta(seconds=1),
        'waiting_on_point_destination': timedelta(seconds=1),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=100),
        'source_point_lat': 0.5,
        'source_point_lon': 0.0,
        'destination_point_lat': 0.5,
        'destination_point_lon': 0.2,
        'waiting_on_point_source': timedelta(seconds=2),
        'waiting_on_point_destination': timedelta(seconds=2),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=100),
        'source_point_lat': 1.0,
        'source_point_lon': 0.0,
        'destination_point_lat': 1.0,
        'destination_point_lon': 0.4,
        'waiting_on_point_source': timedelta(seconds=3),
        'waiting_on_point_destination': timedelta(seconds=3),
    },
]


def test_ctd_simple(tmp_path):
    config_path = tmp_path / 'tmp.json'
    db_path = tmp_path / 'test_db.db'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 10,
            'courier_speed': 0.1
        }
        json.dump(config, f)

    db_logger = DatabaseLogger(run_id=-1)
    reader = DataReader.from_list(TEST_DATA_COURIERS_CTD, TEST_DATA_CLAIMS_CTD, db_logger=db_logger)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=db_logger)
    dsp = HungarianDispatch(DistanceScorer())

    sim.run(dsp, num_iters=5)

    db = Database(db_path)
    db.export_from_logger(db_logger)
    # assert db.get_metric(Metric.CTD) == ((0,), (0,), (0,))
    assert db.get_metric(Metric.CTD, run_id=-1) == approx(16.0, 0.000001)


def test_cumulative_metrics_long_run(tmp_path):
    config_path = Path('configs/simulator.json')
    # db_path = tmp_path / 'test_db.db'

    db_logger = DatabaseLogger(run_id=-1)
    reader = DataReader.from_config(config_path=config_path, sampler_mode='distr_sampler', db_logger=db_logger)
    route_maker = BaseRouteMaker(max_points_lenght=0, cutoff_radius=0.0)  # empty route_maker
    sim = Simulator(data_reader=reader, route_maker=route_maker, config_path=config_path, db_logger=db_logger)
    dsp = HungarianDispatch(DistanceScorer())

    last_completed = 0
    last_cancelled = 0
    last_assigned = 0
    for iter in range(200):
        gamble = sim.get_state()
        assignments = dsp(gamble)
        sim.next(assignments)
        stats = sim.assignment_statistics
        total_completed = sum(1 for item_id, dttm, event, info in db_logger.data['claims'] if event == 'claim_completed')
        total_assigned = sum(1 for item_id, dttm, event, info in db_logger.data['claims'] if event == 'claim_assigned')
        total_cancelled = sum(1 for item_id, dttm, event, info in db_logger.data['claims'] if event == 'claim_cancelled')
        assert total_completed == last_completed + stats['completed_claims']
        assert total_assigned == last_assigned + stats['assigned_not_batched_claims'] + stats['assigned_batched_claims']
        assert total_cancelled == last_cancelled + stats['cancelled_claims']
        last_completed = total_completed
        last_assigned = total_assigned
        last_cancelled = total_cancelled
