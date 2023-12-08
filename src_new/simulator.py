import typing
import json
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict

from .objects import (
    Point,
    Claim,
    Courier,
    Order,
    Gamble,
    Route,
    Assignment,
    CityStamp,
)
from src_new.dispatch import BaseDispatch


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
    def __init__(self, couriers_data, claims_data, start_dttm: datetime) -> None:
        self.couriers_data = couriers_data
        self.claims_data = claims_data
        self.courier_idx = 0
        self.claim_idx = 0
        self.last_dttm = start_dttm

    @staticmethod
    def from_file(claims_path: Path, couriers_path: Path) -> 'DataReader':
        pass

    @staticmethod
    def from_config(config_path: Path) -> 'DataReader':
        pass

    @staticmethod
    def from_list(couriers_data: list[dict[str, typing.Any]], claims_data: list[dict[str, typing.Any]]):
        """
        Given a table (in a form of list of dicts) with 
        """
        # for col in ['start_dttm', 'end_dttm', 'start_position_lat', 'start_position_lon']:
        #     assert col in co
        start_dttm = min(couriers_data[0]['start_dttm'], claims_data[0]['created_dttm'])
        return DataReader(couriers_data, claims_data, start_dttm)

    def get_next_city_stamp(self, timedelta: timedelta) -> CityStamp:
        """
        Provide information about the environment for a given time interval in iterative manner
        """
        if self.courier_idx == len(self.couriers_data) and self.claim_idx == len(self.claims_data):
            raise StopIteration

        start_dttm = self.last_dttm
        stop_dttm = self.last_dttm + timedelta

        couriers: list[Courier] = []
        while self.courier_idx < len(self.couriers_data):
            courier = self.couriers_data[self.courier_idx]
            if courier['start_dttm'] < stop_dttm:
                couriers.append(Courier(
                    id=self.courier_idx,
                    position=Point(courier['start_position_lat'], courier['start_position_lon']),
                    start_dttm=courier['start_dttm'],
                    end_dttm=courier['end_dttm'],
                    courier_type='auto'
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
                    waiting_on_point=claim['waiting_on_point'],
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

        return city_stamp


class Simulator(object):
    """A simulatior of the environment
    """
    def __init__(self, data_reader: DataReader, config_path: Path) -> None:
        self.data_reader = data_reader
        self._config_path = config_path
        self.reset()

    def reset(self) -> None:
        """Resets simulator. Does not reset data_reader!
        """
        with open(self._config_path, 'r') as f:
            config = json.load(f)

        self.active_orders: dict[int, Order] = {}
        self.unassigned_claims: dict[int, Claim] = {}
        self.free_couriers: dict[int, Courier] = {}

        self.gamble_interval: timedelta = timedelta(seconds=config['gamble_duration_interval_sec'])
        self.speed = config['courier_speed']
        # self.statistics = defaultdict(defaultdict(int))

        self._next_order_id = 0
        self._iter = 0
        self._current_gamble_begin_dttm: datetime = datetime.min
        self._current_gamble_end_dttm: datetime = datetime.min

    def next(self, assignments: Assignment) -> None:
        """Makes a step of simulation: assignes new orders, inserts new claims and couriers, drops completed orders, ect

        Args:
            assignments (Assignment): assignments to make
        """
        city_stamp = self.data_reader.get_next_city_stamp(self.gamble_interval)
        self._current_gamble_begin_dttm = city_stamp.from_dttm
        self._current_gamble_end_dttm = city_stamp.to_dttm

        self._assign_active_orders(assignments)
        self._set_new_couriers(city_stamp.couriers)
        self._set_new_claims(city_stamp.claims)

        self._next_free_couriers()
        self._next_unassigned_claims()
        self._next_active_orders()

        self._check_state_validness()
        self._iter += 1

    def get_state(self) -> Gamble:
        """Returns current simulator state in Gamble format

        Returns:
            Gamble: current state
        """
        return Gamble(
            couriers=list(self.free_couriers.values()),
            claims=list(self.unassigned_claims.values()),
            orders=list(self.active_orders.values()),
            dttm_start=self._current_gamble_begin_dttm,
            dttm_end=self._current_gamble_end_dttm
        )

    def run(self, dispatch: BaseDispatch, max_iters: int | None = None) -> None:
        """Runs simulations using a dispatch

        Args:
            dispatch (BaseDispatch): a dispatch using to make assignments
            max_iters (int | None, optional): max iterations of simulations. Defaults to None.
        """
        while True:
            if self._iter >= max_iters:
                return
            gamble = self.get_state()
            assignments = dispatch(gamble)
            try:
                self.next(assignments)
            except StopIteration:
                return
    
    def finish(self) -> None:
        pass

    def get_statistics(self) -> dict[str, list[float]]:
        """Provides simulation statistics

        Returns:
            dict[str, list[float]]: statistics
        """
        return self.statistics

    def _get_next_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def _next_active_orders(self) -> None:
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]
            order.next(self._current_gamble_end_dttm, self.speed)
            if order.done():
                self._update_orders_statistics(order)
                if order.courier.done():
                    self._update_couriers_statistics(order.courier)
                else:
                    self.free_couriers[order.courier.id] = order.courier
                del self.active_orders[order_id]

    def _assign_active_orders(self, assignments: Assignment) -> None:
        for courier_id, claim_id in assignments.ids:
            assert courier_id in self.free_couriers
            assert claim_id in self.unassigned_claims
            claim = self.unassigned_claims[claim_id]
            order = Order(
                id=self._get_next_order_id(),
                creation_dttm=self._current_gamble_begin_dttm,
                courier=self.free_couriers[courier_id],
                route=Route([
                    claim.source_point,
                    claim.destination_point,
                    ]),
                waiting_on_point=claim.waiting_on_point,
                claims=[claim]
            )
            del self.free_couriers[courier_id]
            del self.unassigned_claims[claim_id]
            self.active_orders[order.id] = order

    def _next_free_couriers(self) -> None:
        for courier_id in list(self.free_couriers.keys()):
            courier = self.free_couriers[courier_id]
            courier.next(self._current_gamble_end_dttm)
            if courier.done():
                self._update_couriers_statistics(self.free_couriers[courier_id])
                del self.free_couriers[courier_id]

    def _set_new_couriers(self, new_couriers: list[Courier]) -> None:
        for courier in new_couriers:
            self.free_couriers[courier.id] = courier

    def _next_unassigned_claims(self) -> None:
        for claim_id in list(self.unassigned_claims.keys()):
            claim = self.unassigned_claims[claim_id]
            claim.next(self._current_gamble_end_dttm)
            if claim.done():
                self._update_claims_statistics(claim)
                del self.unassigned_claims[claim_id]

    def _set_new_claims(self, new_claims: list[Claim]) -> None:
        for claim in new_claims:
            self.unassigned_claims[claim.id] = claim

    def _update_orders_statistics(self, order: Order) -> None:
        assert order.done()
        # self.statistics['completed_orders'][self._iter] += 1

    def _update_couriers_statistics(self, courier: Courier) -> None:
        assert courier.done()
        # self.statistics['completed_orders'][self._iter] += 1

    def _update_claims_statistics(self, claim: Claim) -> None:
        assert claim.done()
        pass

    def _check_state_validness(self):
        pass
