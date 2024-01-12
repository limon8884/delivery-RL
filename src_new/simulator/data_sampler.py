import random
import typing
from datetime import datetime, timedelta
from typing import Any
from scipy import stats

from src_new.simulator.utils import CityStamp
from src_new.objects import (
    Point,
    Claim,
    Courier,
)
from src_new.database.logger import Logger
from src_new.utils import get_random_point


class BasePositionSampler():
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        pass

    def sample_courier_start_position(self) -> Point:
        raise NotImplementedError

    def sample_claim_source_point(self) -> Point:
        raise NotImplementedError

    def sample_claim_destination_point(self) -> Point:
        raise NotImplementedError


class BaseDoneDTTMSampler():
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        pass

    def sample_claim_cancell_dttm(self, dttm: datetime) -> datetime:
        raise NotImplementedError

    def sample_courier_end_dttm(self, dttm: datetime) -> datetime:
        raise NotImplementedError


class BaseWaitingTimeSampler():
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        pass

    def sample_waiting_time_on_souce(self) -> timedelta:
        raise NotImplementedError

    def sample_waiting_time_on_destination(self) -> timedelta:
        raise NotImplementedError


class BaseNumSampler:
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        pass

    def sample_num_couriers(self, from_dttm: datetime, to_dttm: datetime) -> int:
        raise NotImplementedError

    def sample_num_claims(self, from_dttm: datetime, to_dttm: datetime) -> int:
        raise NotImplementedError


class DummyPositionSampler(BasePositionSampler):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
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


class DummyDoneDTTMSampler(BaseDoneDTTMSampler):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        self.cancell_claim_timedelta = timedelta(seconds=cfg['cancell_claim_time_secs'])
        self.courier_end_timedelta = timedelta(seconds=cfg['courier_end_time_secs'])

    def sample_claim_cancell_dttm(self, dttm: datetime) -> datetime:
        return dttm + self.cancell_claim_timedelta

    def sample_courier_end_dttm(self, dttm: datetime) -> datetime:
        return dttm + self.courier_end_timedelta


class DummyWaitingTimeSampler(BaseWaitingTimeSampler):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        # self.mode = kwargs['mode']
        self.source_waiting_timedelta = timedelta(seconds=cfg['waiting_time_on_source_secs'])
        self.destination_waiting_timedelta = timedelta(seconds=cfg['waiting_time_on_destination_secs'])

    def sample_waiting_time_on_souce(self) -> timedelta:
        return self.source_waiting_timedelta

    def sample_waiting_time_on_destination(self) -> timedelta:
        return self.destination_waiting_timedelta


class DummyNumSampler(BaseNumSampler):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        self.num_couriers = cfg['num_couriers']
        self.num_claims = cfg['num_claims']

    def sample_num_couriers(self, from_dttm: datetime, to_dttm: datetime) -> int:
        return self.num_couriers

    def sample_num_claims(self, from_dttm: datetime, to_dttm: datetime) -> int:
        return self.num_claims


class DistributionPositionSampler(BasePositionSampler):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.squares: list[tuple[Point, Point]] = []
        self.source_square_probas: list[float] = []
        self.destination_square_probas: list[float] = []
        self.left_lower = Point(cfg['bounds']['left'], cfg['bounds']['lower'])
        self.right_upper = Point(cfg['bounds']['right'], cfg['bounds']['upper'])

        delta_lat = Point(0.0, (cfg['bounds']['upper'] - cfg['bounds']['lower']) / cfg['num_squares_lat'])
        delta_lon = Point((cfg['bounds']['right'] - cfg['bounds']['left']) / cfg['num_squares_lon'], 0.0)
        for i in range(cfg['num_squares_lon']):
            for j in range(cfg['num_squares_lat']):
                self.squares.append((
                    self.left_lower + delta_lon * i + delta_lat * j,
                    self.left_lower + delta_lon * (i + 1) + delta_lat * (j + 1)
                ))
                self.source_square_probas.append(cfg['source_squares_probs'][i][j])
                self.destination_square_probas.append(cfg['destination_squares_probs'][i][j])

    def sample_claim_source_point(self) -> Point:
        square_idx = random.randint(0, len(self.source_square_probas) - 1)
        return get_random_point(self.squares[square_idx])

    def sample_claim_destination_point(self) -> Point:
        square_idx = random.randint(0, len(self.destination_square_probas) - 1)
        return get_random_point(self.squares[square_idx])

    def sample_courier_start_position(self) -> Point:
        return get_random_point((self.left_lower, self.right_upper))


class DistributionDoneDTTMSampler(BaseDoneDTTMSampler):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def sample_claim_cancell_dttm(self, dttm: datetime) -> datetime:
        cfg = self.cfg['cancell_claim_time_secs_distrs']
        distr_idx = random.choices(range(len(cfg)), weights=[d['probability'] for d in cfg])[0]
        distr = getattr(stats, cfg[distr_idx]['distribution'])
        secs = distr.rvs(**(cfg[distr_idx]['params']), size=1)[0]
        return dttm + timedelta(seconds=secs)

    def sample_courier_end_dttm(self, dttm: datetime) -> datetime:
        cfg = self.cfg['courier_end_time_secs_distrs']
        distr_idx = random.choices(range(len(cfg)), weights=[d['probability'] for d in cfg])[0]
        distr = getattr(stats, cfg[distr_idx]['distribution'])
        secs = distr.rvs(**(cfg[distr_idx]['params']), size=1)[0]
        return dttm + timedelta(seconds=secs)


class DistributionWaitingTimeSampler(BaseWaitingTimeSampler):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def sample_waiting_time_on_souce(self) -> datetime:
        cfg = self.cfg['waiting_time_on_source_secs_distr']
        distr = getattr(stats, cfg['distribution'])
        secs = distr.rvs(**cfg['params'], size=1)[0]
        return timedelta(seconds=secs)

    def sample_waiting_time_on_destination(self) -> datetime:
        cfg = self.cfg['waiting_time_on_destination_secs']
        distr = getattr(stats, cfg['distribution'])
        secs = distr.rvs(**cfg['params'], size=1)[0]
        return timedelta(seconds=secs)


class DistributionNumSampler(BaseNumSampler):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def sample_num_claims(self, from_dttm: datetime, to_dttm: datetime) -> int:
        cfg = self.cfg['claims']
        mean_num = cfg['hourly_intensivity'][from_dttm.hour] * cfg['total_daily_num']
        variation = (2 * random.random() - 1) * cfg['epsilon_variation']
        return int(mean_num * variation)

    def sample_num_couriers(self, from_dttm: datetime, to_dttm: datetime) -> int:
        cfg = self.cfg['couriers']
        mean_num = cfg['hourly_intensivity'][from_dttm.hour] * cfg['total_daily_num']
        variation = (2 * random.random() - 1) * cfg['epsilon_variation']
        return int(mean_num * variation)


class CityStampSampler:
    def __init__(self, logger: typing.Optional[Logger], cfg: dict[str, typing.Any]) -> None:
        self._next_claim_id = 0
        self._next_courier_id = 0
        self._logger = logger

        if cfg['sampler_mode'] == 'dummy':
            self._num_sampler = DummyNumSampler(cfg['num_sampler'])
            self._pos_sampler = DummyPositionSampler(cfg['pos_sampler'])
            self._done_dttm_sampler = DummyDoneDTTMSampler(cfg['done_dttm_sampler'])
            self._waiting_time_sampler = DummyWaitingTimeSampler(cfg['waiting_time_sampler'])
        elif cfg['sampler_mode'] == 'distribution':
            self._num_sampler = DistributionNumSampler(cfg['num_sampler'])
            self._pos_sampler = DistributionPositionSampler(cfg['pos_sampler'])
            self._done_dttm_sampler = DistributionDoneDTTMSampler(cfg['done_dttm_sampler'])
            self._waiting_time_sampler = DistributionWaitingTimeSampler(cfg['waiting_time_sampler'])
        else:
            raise RuntimeError('No such sampler mode')

    def sample_citystamp(self, from_dttm: datetime, to_dttm: datetime) -> CityStamp:
        num_couriers = self._num_sampler.sample_num_couriers(from_dttm, to_dttm)
        num_claims = self._num_sampler.sample_num_claims(from_dttm, to_dttm)
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
            logger=self._logger
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
            logger=self._logger
        )
        self._next_courier_id += 1
        return courier
