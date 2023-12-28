import random
import typing
from datetime import datetime, timedelta

from src_new.simulator.utils import CityStamp
from src_new.objects import (
    Point,
    Claim,
    Courier,
)
from src_new.database.database import Database
from src_new.utils import get_random_point


class _PositionSampler:
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        # self.mode = kwargs['mode']
        self.left_lower = Point(
            x=cfg['left_bound'],
            y=cfg['lower_bound'],
        )
        self.right_upper = Point(
            x=cfg['right_bound'],
            y=cfg['upper_bound'],
        )

    def sample_courier_start_position(self) -> Point:
        return get_random_point(corner_bounds=(self.left_lower, self.right_upper))

    def sample_claim_source_point(self) -> Point:
        return get_random_point(corner_bounds=(self.left_lower, self.right_upper))

    def sample_claim_destination_point(self) -> Point:
        return get_random_point(corner_bounds=(self.left_lower, self.right_upper))


class _DoneDTTMSampler:
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        # self.mode = kwargs['mode']
        self.cancell_claim_timedelta = timedelta(seconds=cfg['cancell_claim_time_secs'])
        self.courier_end_timedelta = timedelta(seconds=cfg['courier_end_time_secs'])

    def sample_claim_cancell_dttm(self, dttm: datetime) -> datetime:
        return dttm + self.cancell_claim_timedelta

    def sample_courier_end_dttm(self, dttm: datetime) -> datetime:
        return dttm + self.courier_end_timedelta


class _WaitingTimeSampler:
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        # self.mode = kwargs['mode']
        self.source_waiting_timedelta = timedelta(seconds=cfg['waiting_time_on_source_secs'])
        self.destination_waiting_timedelta = timedelta(seconds=cfg['waiting_time_on_destination_secs'])

    def sample_waiting_time_on_souce(self) -> datetime:
        return self.source_waiting_timedelta

    def sample_waiting_time_on_destination(self) -> datetime:
        return self.destination_waiting_timedelta


class _NumSampler:
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        self.num_couriers = cfg['num_couriers']
        self.num_claims = cfg['num_claims']

    def sample_num_couriers(self) -> int:
        return self.num_couriers

    def sample_num_claims(self) -> int:
        return self.num_claims


class CityStampSampler:
    def __init__(self, db: typing.Optional[Database], cfg: dict[str, typing.Any]) -> None:
        self._next_claim_id = 0
        self._next_courier_id = 0
        self._db = db

        self._num_sampler = _NumSampler(cfg['num_sampler'])
        self._pos_sampler = _PositionSampler(cfg['pos_sampler'])
        self._done_dttm_sampler = _DoneDTTMSampler(cfg['done_dttm_sampler'])
        self._waiting_time_sampler = _WaitingTimeSampler(cfg['waiting_time_sampler'])

    def sample_citystamp(self, from_dttm: datetime, to_dttm: datetime) -> CityStamp:
        num_couriers = self._num_sampler.sample_num_couriers()
        num_claims = self._num_sampler.sample_num_claims()
        claims = [self._sample_claim(self._random_dttm(from_dttm, to_dttm)) for _ in range(num_claims)]
        couriers = [self._sample_courier(self._random_dttm(from_dttm, to_dttm)) for _ in range(num_couriers)]
        return CityStamp(from_dttm, to_dttm, couriers, claims)

    @staticmethod
    def _random_dttm(from_dttm: datetime, to_dttm: datetime) -> datetime:
        total_secs = (to_dttm - from_dttm).total_seconds()
        secs = int(total_secs * random.random())
        return from_dttm + timedelta(seconds=secs)

    def _sample_claim(self, dttm: datetime) -> Claim:
        claim = Claim(
            id=self._next_claim_id,
            source_point=self._pos_sampler.sample_claim_source_point(),
            destination_point=self._pos_sampler.sample_claim_destination_point(),
            creation_dttm=dttm,
            cancell_if_not_assigned_dttm=self._done_dttm_sampler.sample_claim_cancell_dttm(dttm),
            waiting_on_point_source=self._waiting_time_sampler.sample_waiting_time_on_souce(),
            waiting_on_point_destination=self._waiting_time_sampler.sample_waiting_time_on_destination(),
            db=self._db
        )
        self._next_claim_id += 1
        return claim

    def _sample_courier(self, dttm: datetime) -> Courier:
        courier = Courier(
            id=self._next_courier_id,
            position=self._pos_sampler.sample_courier_start_position(),
            start_dttm=dttm,
            end_dttm=self._done_dttm_sampler.sample_courier_end_dttm(dttm),
            courier_type='auto',
            db=self._db
        )
        self._next_courier_id += 1
        return courier
