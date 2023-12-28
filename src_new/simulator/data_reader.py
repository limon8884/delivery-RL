import typing
import json
from pathlib import Path
from datetime import timedelta, datetime
from enum import Enum

from src_new.objects import (
    Point,
    Claim,
    Courier,
)
from src_new.database.database import Database
from src_new.simulator.utils import CityStamp
from src_new.simulator.data_sampler import CityStampSampler


class DataReader:
    """
    A Reader for providing data for simulator in appropriate format.
    All the data stores in 2 tables: `couriers_data` and `claims_data`.
    Format of tables in the following.
    Couriers table description:
    * start_dttm
    * end_dttm
    * start_position_lat
    * start_position_lon
    Claims table description:
    * created_dttm
    * cancelled_dttm
    * source_point_lat
    * source_point_lon
    * destionation_point_lat
    * destionation_point_lon
    * waiting_on_point
    """
    class Mode(Enum):
        LIST = 'list'
        CONFIG = 'config'
        FILE = 'file'

    def __init__(self, mode: Mode, db: typing.Optional[Database] = None, **kwargs) -> None:
        self.mode = mode
        self._db = db
        self.last_dttm: datetime = kwargs['start_dttm']
        if mode is self.Mode.LIST:
            self.couriers_data = kwargs['couriers_data']
            self.claims_data = kwargs['claims_data']
            self.courier_idx = 0
            self.claim_idx = 0
        elif mode is self.Mode.CONFIG:
            self._sampler: CityStampSampler = kwargs['sampler']

    @staticmethod
    def from_file(claims_path: Path, couriers_path: Path) -> 'DataReader':
        raise NotImplementedError

    @staticmethod
    def from_config(config_path: Path, db: typing.Optional[Database] = None) -> 'DataReader':
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        sampler = CityStampSampler(db=db, cfg=cfg['sampler'])
        # num_gambles = cfg['num_gambles']
        # gamble_interval = timedelta(seconds=cfg['gamble_interval_secs'])
        start_dttm = datetime.fromisoformat(cfg['start_dttm'])
        return DataReader(mode=DataReader.Mode.CONFIG, db=db, sampler=sampler, start_dttm=start_dttm)

    @staticmethod
    def from_list(couriers_data: list[dict[str, typing.Any]],
                  claims_data: list[dict[str, typing.Any]], db: typing.Optional[Database] = None):
        """
        Given a table (in a form of list of dicts) with
        """
        start_dttm = min(couriers_data[0]['start_dttm'], claims_data[0]['created_dttm'])
        return DataReader(mode=DataReader.Mode.LIST, db=db,
                          couriers_data=couriers_data, claims_data=claims_data, start_dttm=start_dttm)

    def get_next_city_stamp(self, timedelta: timedelta) -> CityStamp:
        """
        Provide information about the environment for a given time interval in iterative manner
        """
        start_dttm = self.last_dttm
        stop_dttm = self.last_dttm + timedelta

        if self.mode is DataReader.Mode.CONFIG:
            city_stamp = self._sampler.sample_citystamp(from_dttm=start_dttm, to_dttm=stop_dttm)
        elif self.mode is DataReader.Mode.LIST:
            couriers: list[Courier] = []
            while self.courier_idx < len(self.couriers_data):
                courier = self.couriers_data[self.courier_idx]
                if courier['start_dttm'] < stop_dttm:
                    couriers.append(Courier(
                        id=self.courier_idx,
                        position=Point(courier['start_position_lat'], courier['start_position_lon']),
                        start_dttm=courier['start_dttm'],
                        end_dttm=courier['end_dttm'],
                        courier_type='auto',
                        db=self._db
                    ))
                    self.courier_idx += 1
                else:
                    break
            claims: list[Claim] = []
            while self.claim_idx < len(self.claims_data):
                claim = self.claims_data[self.claim_idx]
                if claim['created_dttm'] < stop_dttm:
                    claims.append(Claim(
                        id=self.claim_idx,
                        creation_dttm=claim['created_dttm'],
                        source_point=Point(claim['source_point_lat'], claim['source_point_lon']),
                        destination_point=Point(claim['destination_point_lat'], claim['destination_point_lon']),
                        cancell_if_not_assigned_dttm=claim['cancelled_dttm'],
                        waiting_on_point_source=claim['waiting_on_point_source'],
                        waiting_on_point_destination=claim['waiting_on_point_destination'],
                        db=self._db
                    ))
                    self.claim_idx += 1
                else:
                    break
            city_stamp = CityStamp(
                from_dttm=start_dttm,
                to_dttm=stop_dttm,
                couriers=couriers,
                claims=claims
            )

        self.last_dttm = stop_dttm
        if self._db is not None:
            self._db.commit()

        return city_stamp
