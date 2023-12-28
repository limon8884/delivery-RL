import json
from datetime import datetime, timedelta
import random

from src_new.simulator.simulator import DataReader, Simulator
from src_new.dispatchs.hungarian_dispatch import HungarianDispatch
from src_new.dispatchs.scorers import DistanceScorer
from src_new.database.database import Database
# from src_new.utils import get_random_point


BASE_DTTM = datetime.utcnow()

TEST_DATA_COURIERS = [
    {'start_dttm': BASE_DTTM + timedelta(seconds=0), 'end_dttm': BASE_DTTM + timedelta(seconds=19),
     'start_position_lat': 0.9, 'start_position_lon': 0.3},
    {'start_dttm': BASE_DTTM + timedelta(seconds=5), 'end_dttm': BASE_DTTM + timedelta(seconds=70),
     'start_position_lat': 0.9, 'start_position_lon': 0.3},
    {'start_dttm': BASE_DTTM + timedelta(seconds=5), 'end_dttm': BASE_DTTM + timedelta(seconds=11),
     'start_position_lat': 0.1, 'start_position_lon': 0.2},
    {'start_dttm': BASE_DTTM + timedelta(seconds=15), 'end_dttm': BASE_DTTM + timedelta(seconds=25),
     'start_position_lat': random.random(), 'start_position_lon': random.random()},
    {'start_dttm': BASE_DTTM + timedelta(seconds=25), 'end_dttm': BASE_DTTM + timedelta(seconds=35),
     'start_position_lat': random.random(), 'start_position_lon': random.random()},
    {'start_dttm': BASE_DTTM + timedelta(seconds=50), 'end_dttm': BASE_DTTM + timedelta(seconds=70),
     'start_position_lat': random.random(), 'start_position_lon': random.random()},
]

TEST_DATA_CLAIMS = [
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=0),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=100),
        'source_point_lat': 0.1,
        'source_point_lon': 0.2,
        'destination_point_lat': 1.1,
        'destination_point_lon': 1.2,
        'waiting_on_point_source': timedelta(seconds=2),
        'waiting_on_point_destination': timedelta(seconds=2),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=10),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=20),
        'source_point_lat': 0.9,
        'source_point_lon': 0.3,
        'destination_point_lat': 1.9,
        'destination_point_lon': 1.3,
        'waiting_on_point_source': timedelta(seconds=3),
        'waiting_on_point_destination': timedelta(seconds=3),
    },
    {

        'created_dttm': BASE_DTTM + timedelta(seconds=30),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=50),
        'source_point_lat': 0.0,
        'source_point_lon': 1.0,
        'destination_point_lat': 0.5,
        'destination_point_lon': 0.5,
        'waiting_on_point_source': timedelta(seconds=30),
        'waiting_on_point_destination': timedelta(seconds=30),
    },
    {
        'created_dttm': BASE_DTTM + timedelta(seconds=50),
        'cancelled_dttm': BASE_DTTM + timedelta(seconds=70),
        'source_point_lat': 0.0,
        'source_point_lon': 1.0,
        'destination_point_lat': 0.3,
        'destination_point_lon': 0.4,
        'waiting_on_point_source': timedelta(seconds=5),
        'waiting_on_point_destination': timedelta(seconds=5),
    },
]


def test_data_reader_from_list():
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS)
    couriers_nums = []
    claims_nums = []
    for _ in range(6):
        city_stamp = reader.get_next_city_stamp(timedelta(seconds=10))
        couriers_nums.append(len(city_stamp.couriers))
        claims_nums.append(len(city_stamp.claims))
    assert couriers_nums == [3, 1, 1, 0, 0, 1]
    assert claims_nums == [1, 1, 0, 1, 0, 1]


def test_simulator_step_by_step(tmp_path):
    config_path = tmp_path / 'tmp.json'
    # path_db = tmp_path / 'test_db.db'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 10,
            'courier_speed': 0.1
        }
        json.dump(config, f)
    reader = DataReader.from_list(TEST_DATA_COURIERS, TEST_DATA_CLAIMS, db=None)
    sim = Simulator(data_reader=reader, config_path=config_path, db=None)
    dsp = HungarianDispatch(DistanceScorer())

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (0, 0, 0)
    assignments = dsp(last_gamble)
    assert assignments.ids == []
    sim.next(assignments)

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (3, 1, 0)
    assignments = dsp(last_gamble)
    assert sorted(assignments.ids) == [(2, 0)]
    sim.next(assignments)

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (2, 1, 1)
    assignments = dsp(last_gamble)
    assert sorted(assignments.ids) == [(1, 1)]
    sim.next(assignments)

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (1, 0, 1)
    assert last_gamble.couriers[0].id == 4
    assert last_gamble.orders[0].id == 1
    assignments = dsp(last_gamble)
    assert sorted(assignments.ids) == []
    sim.next(assignments)

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (0, 1, 1)
    assert last_gamble.claims[0].id == 2
    assert last_gamble.orders[0].id == 1
    assignments = dsp(last_gamble)
    assert sorted(assignments.ids) == []
    sim.next(assignments)

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (1, 1, 0)
    assert last_gamble.claims[0].id == 2
    assert last_gamble.couriers[0].id == 1
    assignments = dsp(last_gamble)
    assert sorted(assignments.ids) == [(1, 2)]
    sim.next(assignments)

    last_gamble = sim.get_state()
    assert (len(last_gamble.couriers), len(last_gamble.claims), len(last_gamble.orders)) == (1, 1, 1)
    assert last_gamble.couriers[0].id == 5
    assert last_gamble.claims[0].id == 3
    assert last_gamble.orders[0].id == 2
    assignments = dsp(last_gamble)
    assert sorted(assignments.ids) == [(5, 3)]


def test_simulator_full_run_wo_db(tmp_path):
    n_iters = 1000
    config_path = tmp_path / 'tmp.json'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 10,
            'courier_speed': 0.1
        }
        json.dump(config, f)

    couriers = [
        {
            'start_dttm': BASE_DTTM + timedelta(seconds=i),
            'end_dttm': BASE_DTTM + timedelta(seconds=i + 1),
            'start_position_lat': random.random(),
            'start_position_lon': random.random()
        }
        for i in range(n_iters)
    ]
    claims = [
        {
            'created_dttm': BASE_DTTM + timedelta(seconds=i),
            'cancelled_dttm': BASE_DTTM + timedelta(seconds=i + 1),
            'source_point_lat': random.random(),
            'source_point_lon': random.random(),
            'destination_point_lat': random.random(),
            'destination_point_lon': random.random(),
            'waiting_on_point_source': timedelta(seconds=0),
            'waiting_on_point_destination': timedelta(seconds=0),
        }
        for i in range(n_iters)
    ]

    reader = DataReader.from_list(couriers, claims, db=None)
    sim = Simulator(data_reader=reader, config_path=config_path, db=None)
    dsp = HungarianDispatch(DistanceScorer())

    sim.run(dsp, num_iters=n_iters)


def test_simulator_full_run_with_db(tmp_path):
    n_iters = 100
    config_path = tmp_path / 'tmp.json'
    db_path = tmp_path / 'test_db.db'
    with open(config_path, 'w') as f:
        config = {
            'gamble_duration_interval_sec': 10,
            'courier_speed': 0.1
        }
        json.dump(config, f)

    couriers = [
        {
            'start_dttm': BASE_DTTM + timedelta(seconds=i),
            'end_dttm': BASE_DTTM + timedelta(seconds=i + 1),
            'start_position_lat': random.random(),
            'start_position_lon': random.random()
        }
        for i in range(n_iters)
    ]
    claims = [
        {
            'created_dttm': BASE_DTTM + timedelta(seconds=i),
            'cancelled_dttm': BASE_DTTM + timedelta(seconds=i + 1),
            'source_point_lat': random.random(),
            'source_point_lon': random.random(),
            'destination_point_lat': random.random(),
            'destination_point_lon': random.random(),
            'waiting_on_point_source': timedelta(seconds=0),
            'waiting_on_point_destination': timedelta(seconds=0),
        }
        for i in range(n_iters)
    ]

    db = Database(db_path)
    reader = DataReader.from_list(couriers, claims, db=db)
    sim = Simulator(data_reader=reader, config_path=config_path, db=db)
    dsp = HungarianDispatch(DistanceScorer())

    sim.run(dsp, num_iters=n_iters)
