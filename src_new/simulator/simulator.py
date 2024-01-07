import typing as tp
import json
from pathlib import Path
from datetime import timedelta, datetime
from tqdm import tqdm

from src_new.objects import (
    Claim,
    Courier,
    Order,
    Gamble,
    Route,
    Assignment,
)
from src_new.dispatchs.base_dispatch import BaseDispatch
from src_new.simulator.data_reader import DataReader
from src_new.router_makers import BaseRouteMaker
from src_new.database.logger import Logger


class Simulator(object):
    """A simulatior of the environment
    """
    def __init__(self, data_reader: DataReader, route_maker: BaseRouteMaker,
                 config_path: Path, logger: tp.Optional[Logger] = None) -> None:
        self.route_maker = route_maker
        self.data_reader = data_reader
        self._config_path = config_path
        self._logger = logger
        self.reset()

    def reset(self) -> None:
        """Resets simulator. Does not reset data_reader!
        """
        with open(self._config_path, 'r') as f:
            config = json.load(f)

        self.active_orders: dict[int, Order] = {}
        self.unassigned_claims: dict[int, Claim] = {}
        self.free_couriers: dict[int, Courier] = {}
        self.courier_id_to_order_id: dict[int, int] = {}

        self.gamble_interval: timedelta = timedelta(seconds=config['gamble_duration_interval_sec'])
        self.speed = config['courier_speed']

        self._next_order_id = 0
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

    def run(self, dispatch: BaseDispatch, num_iters: int) -> None:
        """Runs simulations using a dispatch

        Args:
            dispatch (BaseDispatch): a dispatch using to make assignments
            max_iters (int | None, optional): max iterations of simulations. Defaults to None.
        """
        for iter in tqdm(range(num_iters)):
            gamble = self.get_state()
            assignments = dispatch(gamble)
            self.next(assignments)

    def _get_next_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def _next_active_orders(self) -> None:
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]
            order.next(self._current_gamble_end_dttm, self.speed)
            if order.done():
                del self.courier_id_to_order_id[order.courier.id]
                if not order.courier.done():
                    self.free_couriers[order.courier.id] = order.courier
                del self.active_orders[order_id]

    def _assign_active_orders(self, assignments: Assignment) -> None:
        for courier_id, claim_id in assignments.ids:
            assert claim_id in self.unassigned_claims
            claim = self.unassigned_claims[claim_id]
            if courier_id in self.free_couriers:
                order = Order(
                    id=self._get_next_order_id(),
                    creation_dttm=self._current_gamble_begin_dttm,
                    courier=self.free_couriers[courier_id],
                    route=Route.from_claim(claim),
                    claims=[claim],
                    logger=self._logger
                )
                del self.free_couriers[courier_id]
                self.active_orders[order.id] = order
                self.courier_id_to_order_id[courier_id] = order.id
            else:
                assert courier_id in self.courier_id_to_order_id
                order = self.active_orders[self.courier_id_to_order_id[courier_id]]
                self.route_maker.add_claim(order.route, order.courier.position, claim)
            del self.unassigned_claims[claim_id]

    def _next_free_couriers(self) -> None:
        for courier_id in list(self.free_couriers.keys()):
            courier = self.free_couriers[courier_id]
            assert not courier.done()
            courier.next(self._current_gamble_end_dttm)
            if courier.done():
                del self.free_couriers[courier_id]

    def _set_new_couriers(self, new_couriers: list[Courier]) -> None:
        for courier in new_couriers:
            self.free_couriers[courier.id] = courier

    def _next_unassigned_claims(self) -> None:
        for claim_id in list(self.unassigned_claims.keys()):
            claim = self.unassigned_claims[claim_id]
            assert not claim.done()
            claim.next(self._current_gamble_end_dttm)
            if claim.done():
                del self.unassigned_claims[claim_id]

    def _set_new_claims(self, new_claims: list[Claim]) -> None:
        for claim in new_claims:
            self.unassigned_claims[claim.id] = claim
