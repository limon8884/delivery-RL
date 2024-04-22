import typing as tp
import json
import logging
import warnings
from pathlib import Path
from datetime import timedelta, datetime
from tqdm import tqdm
from collections import defaultdict

from src.objects import (
    Claim,
    Courier,
    Order,
    Gamble,
    Route,
    Assignment,
    Point,
)
from src.visualization import Visualization
from src.dispatchs.base_dispatch import BaseDispatch
from src.simulator.data_reader import DataReader
from src.router_makers import BaseRouteMaker
from src.database.logger import DatabaseLogger

logging.basicConfig(filename='running_logs.log', encoding='utf-8', level=logging.WARNING)
LOGGER = logging.getLogger(__name__)


class Simulator(object):
    """A simulatior of the environment
    """
    def __init__(self, data_reader: DataReader, route_maker: BaseRouteMaker,
                 config_path: Path, db_logger: tp.Optional[DatabaseLogger] = None) -> None:
        self.route_maker = route_maker
        self.data_reader = data_reader
        self._config_path = config_path
        self._logger = db_logger
        self.reset()

    def _update_assignment_statistics(self) -> None:
        # Additive statistics
        self.assignment_statistics = {
            'new_couriers': 0.0,
            'new_claims': 0.0,
            'assigned_couriers': 0.0,
            'assigned_not_batched_claims': 0.0,
            'assigned_batched_claims': 0.0,
            'assigned_not_batched_orders_arrival_distance': 0.0,
            'assigned_batched_orders_distance_increase': 0.0,
            'prohibited_assignments': 0.0,
            'finished_couriers': 0.0,
            'cancelled_claims': 0.0,
            'completed_claims': 0.0,
            'completed_orders': 0.0,
        }

    def reset(self) -> None:
        """Resets simulator. Does not reset data_reader!
        """
        if self._logger is not None:
            self._logger.reset()

        with open(self._config_path, 'r') as f:
            config = json.load(f)

        self._update_assignment_statistics()
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
        self._update_assignment_statistics()

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

    def run(self, dispatch: BaseDispatch, num_iters: int,
            visualization: Visualization | None = None) -> None:
        """Runs simulations using a dispatch

        Args:
            dispatch (BaseDispatch): a dispatch using to make assignments
            max_iters (int | None, optional): max iterations of simulations. Defaults to None.
        """
        for iter in tqdm(range(num_iters)):
            gamble = self.get_state()
            assignments = dispatch(gamble)
            self.next(assignments)
            if visualization is not None:
                visualization.visualize(gamble, assignments, iter)

    def _get_next_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id

    def _next_active_orders(self) -> None:
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]
            order.next(self._current_gamble_end_dttm, speed=self.speed)
            if order.done():
                del self.courier_id_to_order_id[order.courier.id]
                if not order.courier.done():
                    self.free_couriers[order.courier.id] = order.courier
                else:
                    self.assignment_statistics['finished_couriers'] += 1
                del self.active_orders[order_id]
                self.assignment_statistics['completed_claims'] += len(order.claims)
                self.assignment_statistics['completed_orders'] += 1

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
                self.assignment_statistics['assigned_not_batched_claims'] += 1
                self.assignment_statistics['assigned_couriers'] += 1
                self.assignment_statistics['assigned_not_batched_orders_arrival_distance'] += \
                    order.route.distance_of_courier_arrival(order.courier.position)
            else:
                assert courier_id in self.courier_id_to_order_id
                order = self.active_orders[self.courier_id_to_order_id[courier_id]]
                if len(order.route.route_points) > self.route_maker.max_points_lenght - 2:
                    warnings.warn('assigned on order with to many points')
                    self.assignment_statistics['prohibited_assignments'] += 1
                    continue
                if not order.courier.is_time_off():
                    prev_route_dist = order.route.distance_with_arrival(order.courier.position)
                    self.route_maker.add_claim(order.route, order.courier.position, claim)
                    order.claims[claim_id] = claim
                    claim.assign()
                    self.assignment_statistics['assigned_batched_claims'] += 1
                    self.assignment_statistics['assigned_batched_orders_distance_increase'] += \
                        order.route.distance_with_arrival(order.courier.position) - prev_route_dist
            del self.unassigned_claims[claim_id]

    def _next_free_couriers(self) -> None:
        for courier_id in list(self.free_couriers.keys()):
            courier = self.free_couriers[courier_id]
            assert not courier.done()
            courier.next(self._current_gamble_end_dttm)
            if courier.done():
                del self.free_couriers[courier_id]
                self.assignment_statistics['finished_couriers'] += 1

    def _set_new_couriers(self, new_couriers: list[Courier]) -> None:
        for courier in new_couriers:
            self.free_couriers[courier.id] = courier
        self.assignment_statistics['new_couriers'] += len(new_couriers)

    def _next_unassigned_claims(self) -> None:
        for claim_id in list(self.unassigned_claims.keys()):
            claim = self.unassigned_claims[claim_id]
            assert not claim.done()
            claim.next(self._current_gamble_end_dttm)
            if claim.done():
                del self.unassigned_claims[claim_id]
                self.assignment_statistics['cancelled_claims'] += 1

    def _set_new_claims(self, new_claims: list[Claim]) -> None:
        for claim in new_claims:
            self.unassigned_claims[claim.id] = claim
        self.assignment_statistics['new_claims'] += len(new_claims)


class ConstantSimulator(Simulator):
    def __init__(self, config_path: Path) -> None:
        self.assignment_statistics: dict[str, float] = defaultdict(float)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self._dttm = datetime.min
        self._create_gamble()
        self.reset()

    def _create_gamble(self) -> Gamble:
        self._gamble = Gamble(
            couriers=[self._make_courier(data) for data in self.config['couriers']],
            claims=[self._make_claim(data) for data in self.config['claims']],
            orders=[self._make_order(data) for data in self.config['orders']],
            dttm_start=self._dttm,
            dttm_end=self._dttm + timedelta(seconds=30),
        )

    def _make_courier(self, cfg: dict[str, tp.Any]) -> Courier:
        return Courier(
            id=cfg['id'],
            position=Point(cfg['position']['x'], cfg['position']['y']),
            start_dttm=self._dttm,
            end_dttm=self._dttm,
            courier_type='auto',
        )

    def _make_claim(self, cfg: dict[str, tp.Any]) -> Claim:
        return Claim(
            id=cfg['id'],
            source_point=Point(cfg['source']['x'], cfg['source']['y']),
            destination_point=Point(cfg['destination']['x'], cfg['destination']['y']),
            creation_dttm=self._dttm,
            cancell_if_not_assigned_dttm=self._dttm + timedelta(seconds=100),
            waiting_on_point_source=timedelta(seconds=10),
            waiting_on_point_destination=timedelta(seconds=10),
        )

    def _make_order(self, cfg: dict[str, tp.Any]) -> Order:
        claim = self._make_claim(cfg['claim'])
        return Order(
            id=cfg['id'],
            creation_dttm=datetime.min,
            courier=self._make_courier(cfg['courier']),
            route=Route.from_claim(claim),
            claims=[claim],
        )

    def reset(self) -> None:
        pass

    def get_state(self) -> Gamble:
        return self._gamble

    def next(self, assignments: Assignment) -> None:
        self.assignment_statistics = defaultdict(float)
        crr_ids = {crr.id for crr in self._gamble.couriers}
        crr_ord_ids = {ord.courier.id for ord in self._gamble.orders}
        clm_ids = {clm.id for clm in self._gamble.claims}
        for crr_id, clm_id in assignments.ids:
            assert clm_id in clm_ids, (clm_id, clm_ids)
            if crr_id in crr_ids:
                crr_ids.remove(crr_id)
                clm_ids.remove(clm_id)
                self.assignment_statistics['assigned_not_batched_claims'] += 1
            elif crr_id in crr_ord_ids:
                crr_ord_ids.remove(crr_id)
                clm_ids.remove(clm_id)
                self.assignment_statistics['assigned_batched_claims'] + 1
            else:
                print(crr_id, crr_ids, crr_ord_ids)
                raise RuntimeError
        self.assignment_statistics['unassigned_couriers'] = len(crr_ids)
        self.assignment_statistics['unassigned_claims'] = len(clm_ids)
        self.assignment_statistics['completed_claims'] = 0
        self.assignment_statistics['cancelled_claims'] = 0
