from datetime import datetime, timedelta

from src.simulator.data_sampler import CityStampSampler


BASE_DTTM = datetime.utcnow()


CFG_DUMMY = {
    'sampler_mode': 'dummy',
    'num_sampler': {
        'num_couriers': 3,
        'num_claims': 5,
    },
    'pos_sampler': {
        'left_bound': 37.35,
        'lower_bound': 55.58,
        'right_bound': 37.85,
        'upper_bound': 55.92,
    },
    'done_dttm_sampler': {
        'cancell_claim_time_secs': 5 * 60,
        'courier_end_time_secs': 5 * 60 * 60,
    },
    'waiting_time_sampler': {
        'waiting_time_on_source_secs': 5 * 60,
        'waiting_time_on_destination_secs': 3 * 60,
    }
}


CFG_DISTR = {
    "sampler_mode": "distribution",
    "num_sampler": {
        "couriers": {
            "epsilon_variation": 0.2,
            "total_daily_num": 24 * 60 * 20,
            "hourly_intensivity": [i / 24 / 23 * 2 for i in range(24)]
        },
        "claims": {
            "epsilon_variation": 0.1,
            "total_daily_num": 60 * 24 * 10,
            "hourly_intensivity": [i / 24 / 23 * 2 for i in reversed(range(24))]
        }
    },
    "pos_sampler": {
        "bounds": {
            "left": 37.4,
            "lower": 55.6,
            "right": 37.8,
            "upper": 55.9
        },
        "num_squares_lat": 5,
        "num_squares_lon": 5,
        "source_squares_probs": [
            [0, 0, 0, 0, 0],
            [0, 0.1, 0.1, 0.1, 0],
            [0, 0.1, 0.2, 0.1, 0],
            [0, 0.1, 0.1, 0.1, 0],
            [0, 0, 0, 0, 0]
        ],
        "destination_squares_probs": [
            [0, 0, 0, 0, 0],
            [0, 0.1, 0.1, 0.1, 0],
            [0, 0.1, 0.2, 0.1, 0],
            [0, 0.1, 0.1, 0.1, 0],
            [0, 0, 0, 0, 0]
        ],
    },
    "done_dttm_sampler": {
        "cancell_claim_time_secs_distrs": [
            {
                "distribution": "expon",
                "probability": 1.0,
                "params": {
                    "loc": 1.0,
                    "scale": 1000.0
                }
            }
        ],
        "courier_end_time_secs_distrs": [
            {
                "distribution": "lognorm",
                "probability": 0.5,
                "params": {
                    "s": 0.9,
                    "loc": -1800,
                    "scale": 9000
                }
            },
            {
                "distribution": "norm",
                "probability": 0.5,
                "params": {
                    "loc": 35000,
                    "scale": 12000
                }
            }
        ]
    },
    "waiting_time_sampler": {
        "waiting_time_on_source_secs_distr": {
            "distribution": "gamma",
            "params": {
                "a": 0.91,
                "loc": 1.0,
                "scale": 492.0
            }
        },
        "waiting_time_on_destination_secs": {
            "distribution": "gamma",
            "params": {
                "a": 1.76,
                "loc": -8.8,
                "scale": 127.2
            }
        }
    }
}


def test_dummy_sampler():
    smp = CityStampSampler(logger=None, cfg=CFG_DUMMY)
    smp.sample_citystamp(BASE_DTTM, BASE_DTTM + timedelta(seconds=30))

    n_stamps = 50
    _ = [
        smp.sample_citystamp(BASE_DTTM + timedelta(seconds=i * 30), BASE_DTTM + timedelta(seconds=30 + i * 30))
        for i in range(n_stamps)
    ]


def test_distribution_sampler():
    smp = CityStampSampler(logger=None, cfg=CFG_DISTR)
    smp.sample_citystamp(BASE_DTTM, BASE_DTTM + timedelta(seconds=30))


def test_dummy_sampler_full():
    smp = CityStampSampler(logger=None, cfg=CFG_DUMMY)
    n_stamps = 10
    city_stamps = [
        smp.sample_citystamp(BASE_DTTM + timedelta(seconds=i * 30), BASE_DTTM + timedelta(seconds=30 + i * 30))
        for i in range(n_stamps)
    ]

    assert len(city_stamps) == n_stamps
    for city_stamp in city_stamps:
        assert len(city_stamp.claims) == 5
        assert len(city_stamp.couriers) == 3
        for claim in city_stamp.claims:
            assert claim.waiting_on_point_source.total_seconds() == 5 * 60
            assert claim.waiting_on_point_destination.total_seconds() == 3 * 60
            assert (claim.cancell_if_not_assigned_dttm - claim.creation_dttm).total_seconds() == 5 * 60
        for courier in city_stamp.couriers:
            assert (courier.end_dttm - courier.start_dttm).total_seconds() == 5 * 60 * 60
            assert courier.position.x < 37.85 and courier.position.x >= 37.35
            assert courier.position.y < 55.92 and courier.position.y >= 55.58
