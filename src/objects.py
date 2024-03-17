import typing
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque


from src.database.classes import TableName, Event
from src.database.logger import Logger


class Point:
    def __init__(self, x: float, y: float) -> None:
        assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(y, float) or isinstance(y, int)
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> 'Point':
        return Point(self.x * other, self.y * other)

    def __rtruediv__(self, other: float) -> 'Point':
        assert other != 0
        return Point(self.x / other, self.y / other)

    def normalize(self) -> 'Point':
        n = (self.x**2 + self.y**2)**0.5
        if n == 0:
            return Point(self.x, self.y)
        return Point(self.x / n, self.y / n)

    def __repr__(self) -> str:
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y

    @staticmethod
    def distance(first_point: 'Point', second_point: 'Point') -> float:
        return ((first_point.x - second_point.x)**2 + (first_point.y - second_point.y)**2)**0.5


class Route:
    class PointType(Enum):
        SOURCE = 0
        DESTINATION = 1

    @dataclass
    class RoutePoint:
        point: Point
        claim_id: int
        point_type: 'Route.PointType'

    def __init__(self, route_points: list[RoutePoint]) -> None:
        self.route_points: deque[Route.RoutePoint] = deque(route_points)
        # self._next_point_idx = 0

    @staticmethod
    def from_points(
        points: list[Point],
        claim_ids: list[int],
        point_types: list['PointType']
    ) -> 'Route':
        route = Route([])
        for point, claim_id, point_type in zip(points, claim_ids, point_types):
            route.route_points.append(Route.RoutePoint(point, claim_id, point_type))
        return route

    @staticmethod
    def from_claim(claim: 'Claim') -> 'Route':
        return Route([
            Route.RoutePoint(claim.source_point, claim.id, Route.PointType.SOURCE),
            Route.RoutePoint(claim.destination_point, claim.id, Route.PointType.DESTINATION),
        ])

    def next(self) -> None:
        assert not self.done()
        self.route_points.popleft()
        # self._next_point_idx += 1

    def done(self):
        return len(self.route_points) == 0
        # return self._next_point_idx == len(self.route_points)

    def next_route_point(self) -> RoutePoint:
        assert not self.done()
        return self.route_points[0]
        # return self.route_points[self._next_point_idx]

    def distance(self) -> float:
        assert not self.done()
        total_dist: float = 0
        current_point = self.route_points[0]
        for i, next_point in enumerate(self.route_points):
            if i == 0:
                continue
            total_dist += Point.distance(current_point.point, next_point.point)
            current_point = next_point
        return total_dist

    def distance_of_courier_arrival(self, courier_position: Point):
        return Point.distance(self.next_route_point().point, courier_position)

    def distance_with_arrival(self, courier_position: Point):
        return self.distance() + self.distance_of_courier_arrival(courier_position)


class Item:
    """
    A base class for Courier, Claim and Order
    """
    def __init__(self, id: int, logger: typing.Optional[Logger]) -> None:
        """
        id - id of item
        dttm - creation datetime in the system
        """
        self.id: int = id
        self._dttm: typing.Optional[datetime] = None
        self._logger: typing.Optional[Logger] = logger

    def next(self, current_time: datetime) -> None:
        """
        Shifts the time of item forward
        If done, raises an Exception
        """
        self._dttm = current_time

    def done(self) -> bool:
        '''
        Checks if done
        '''
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        '''
        Makes a numpy view of a class
        '''
        raise NotImplementedError

    @staticmethod
    def numpy_feature_types(*args, **kwargs) -> dict[tuple[int, int], str]:
        '''
        Describes on which position of numpy view of this class which types of numbers are encoded
        Returns a dict of elements, where key is [from_idx, to_idx) and value is either `number` or `coord`
        '''
        raise NotImplementedError


class Courier(Item):
    class Status(Enum):
        FREE = 0
        PROCESS = 1
        OFFLINE = 2

    def __init__(self,
                 id: int,
                 position: Point,
                 start_dttm: datetime,
                 end_dttm: datetime,
                 courier_type: str,
                 logger: typing.Optional[Logger] = None,
                 ) -> None:
        super().__init__(id, logger)
        self._dttm = start_dttm
        self.position = position
        self.start_dttm = start_dttm
        self.end_dttm = end_dttm
        self.courier_type = courier_type

        self.status: Courier.Status = Courier.Status.FREE
        if self._logger is not None:
            self._logger.insert(TableName.COURIER_TABLE, id, start_dttm, Event.COURIER_STARTED)

    def set_position(self, position: Point) -> None:
        self.position = position

    def is_time_off(self) -> bool:
        return self.end_dttm <= self._dttm

    def next(self, current_time: datetime) -> None:
        if self.done():
            raise RuntimeError("Next courier when done")
        super().next(current_time)
        if self.is_time_off() and self.status is Courier.Status.FREE:
            self.status = Courier.Status.OFFLINE
            if self._logger is not None:
                self._logger.insert(TableName.COURIER_TABLE, self.id, self._dttm, Event.COURIER_ENDED)

    def done(self) -> bool:
        return self.status is Courier.Status.OFFLINE

    def to_numpy(self) -> np.ndarray:
        status_num = self.status.value
        online_secs_num = (self._dttm - self.start_dttm).total_seconds()
        return np.array([self.position.x, self.position.y, status_num, online_secs_num])

    @staticmethod
    def numpy_feature_types() -> dict[tuple[int, int], str]:
        return {
            (0, 2): 'coords',
            (2, 4): 'numbers',
        }


class Claim(Item):
    class Status(Enum):
        UNASSIGNED = 0
        ASSIGNED = 1
        COMPLETED = 2
        CANCELLED = 3

    def __init__(self,
                 id: int,
                 source_point: Point,
                 destination_point: Point,
                 creation_dttm: datetime,
                 cancell_if_not_assigned_dttm: datetime,
                 waiting_on_point_source: timedelta,
                 waiting_on_point_destination: timedelta,
                 logger: typing.Optional[Logger] = None,
                 ) -> None:
        super().__init__(id, logger)
        self._dttm = creation_dttm
        self.source_point = source_point
        self.destination_point = destination_point
        self.creation_dttm = creation_dttm
        self.cancell_if_not_assigned_dttm = cancell_if_not_assigned_dttm
        self.waiting_on_point_source = waiting_on_point_source
        self.waiting_on_point_destination = waiting_on_point_destination

        self.status: Claim.Status = Claim.Status.UNASSIGNED
        if self._logger is not None:
            self._logger.insert(TableName.CLAIM_TABLE, id, creation_dttm, Event.CLAIM_CREATED)

    def next(self, current_time: datetime) -> None:
        if self.done():
            raise RuntimeError('Claim next when done')
        super().next(current_time)
        if self.cancell_if_not_assigned_dttm < self._dttm and self.status is Claim.Status.UNASSIGNED:
            self.cancell()

    def cancell(self):
        self.status = Claim.Status.CANCELLED
        if self._logger is not None:
            self._logger.insert(TableName.CLAIM_TABLE, self.id, self._dttm, Event.CLAIM_CANCELLED)

    def assign(self):
        assert self.status is Claim.Status.UNASSIGNED
        self.status = Claim.Status.ASSIGNED
        if self._logger is not None:
            self._logger.insert(TableName.CLAIM_TABLE, self.id, self._dttm, Event.CLAIM_ASSIGNED)

    def complete(self, seconds_to_act: int):
        assert self.status is Claim.Status.ASSIGNED, (self.status.name, self.id, self._dttm)
        self.status = Claim.Status.COMPLETED
        if self._logger is not None:
            self._logger.insert(TableName.CLAIM_TABLE, self.id,
                                self._dttm - timedelta(seconds=seconds_to_act), Event.CLAIM_COMPLETED)

    def done(self) -> bool:
        return self.status in (Claim.Status.COMPLETED, Claim.Status.CANCELLED)

    def to_numpy(self) -> np.ndarray:
        status_num = self.status.value
        online_secs_num = (self._dttm - self.creation_dttm).total_seconds()
        return np.array([
            self.source_point.x,
            self.source_point.y,
            self.destination_point.x,
            self.destination_point.y,
            status_num,
            online_secs_num
        ])

    @staticmethod
    def numpy_feature_types() -> dict[tuple[int, int], str]:
        return {
            (0, 4): 'coords',
            (4, 6): 'numbers',
        }


class Order(Item):
    class Status(Enum):
        IN_ARRIVAL = 0
        IN_PROCESS = 1
        COMPLETED = 2

    def __init__(self,
                 id: int,
                 creation_dttm: datetime,
                 courier: Courier,
                 route: Route,
                 claims: list[Claim],
                 logger: typing.Optional[Logger] = None,
                 ) -> None:
        super().__init__(id, logger)
        self.creation_dttm = creation_dttm
        self.courier = courier
        self.claims = {claim.id: claim for claim in claims}
        self.route = route

        for claim in claims:
            claim.assign()
        self.status = Order.Status.IN_ARRIVAL
        self.courier.status = Courier.Status.PROCESS
        if self._logger is not None:
            self._logger.insert(TableName.ORDER_TABLE, id, self.creation_dttm, Event.ORDER_CREATED)

        self._dttm = creation_dttm
        self._seconds_to_wait = None

    def done(self) -> bool:
        return self.status is Order.Status.COMPLETED

    def get_rest_distance(self) -> float:
        return self.route.distance_with_arrival(self.courier.position)

    def next(self, current_time: datetime, speed: float) -> None:
        if self.done():
            raise RuntimeError('next called for order when done')
        seconds_to_act = (current_time - self._dttm).total_seconds()
        super().next(current_time)
        self.courier.next(current_time)
        for claim in self.claims.values():
            if not claim.done():
                claim.next(current_time)
        while seconds_to_act > 0:
            if self.route.done():
                self._finish_order(seconds_to_act)
                return
            seconds_to_act = self._move_courier(seconds_to_act, speed)
            seconds_to_act = self._wait_on_point_func(seconds_to_act)

    def _finish_order(self, seconds_to_act: int) -> None:
        self.status = Order.Status.COMPLETED
        self.courier.status = Courier.Status.OFFLINE if self.courier.is_time_off() else Courier.Status.FREE
        if self._logger is not None:
            self._logger.insert(TableName.ORDER_TABLE, self.id,
                                self._dttm - timedelta(seconds=seconds_to_act), Event.ORDER_FINISHED)

    def _wait_on_point_func(self, seconds_to_act: int) -> int:
        '''
        Returns remeining seconds to act
        '''
        if self._seconds_to_wait is None:
            return seconds_to_act
        if self._seconds_to_wait >= seconds_to_act:
            self._seconds_to_wait -= seconds_to_act
            return 0
        seconds_to_act -= self._seconds_to_wait
        self._seconds_to_wait = None
        if self.route.next_route_point().point_type is Route.PointType.DESTINATION:
            self.claims[self.route.next_route_point().claim_id].complete(seconds_to_act)
        self.route.next()
        return seconds_to_act

    def _move_courier(self, seconds_to_act: int, speed: float) -> int:
        '''
        Returns remeining seconds to act
        '''
        if self._seconds_to_wait is not None:
            return seconds_to_act
        next_point = self.route.next_route_point().point
        dist_to_next_point = Point.distance(self.courier.position, next_point)
        if dist_to_next_point > speed * seconds_to_act:
            new_courier_pos = self.courier.position + \
                (next_point - self.courier.position).normalize() * speed * seconds_to_act
            self.courier.set_position(new_courier_pos)
            return 0
        seconds_to_act -= dist_to_next_point / speed
        self._seconds_to_wait = \
            self.claims[self.route.next_route_point().claim_id].waiting_on_point_source.total_seconds() \
            if self.route.next_route_point().point_type is Route.PointType.SOURCE \
            else self.claims[self.route.next_route_point().claim_id].waiting_on_point_destination.total_seconds()
        self.courier.set_position(next_point)
        self.status = Order.Status.IN_PROCESS
        return seconds_to_act

    def to_numpy(self, max_num_points_in_route: int) -> np.ndarray:
        crr_tens = self.courier.to_numpy()
        status_num = self.status.value
        online_secs_num = (self._dttm - self.creation_dttm).total_seconds()
        assert len(self.route.route_points) <= max_num_points_in_route, len(self.route.route_points)
        point_coords = []
        for point_idx in range(max_num_points_in_route):
            if point_idx < len(self.route.route_points):
                point_coords.append(self.route.route_points[point_idx].point.x)
                point_coords.append(self.route.route_points[point_idx].point.y)
            else:
                point_coords.append(0)
                point_coords.append(0)

        pts_tens = np.array([status_num, online_secs_num] + point_coords)
        return np.concatenate([crr_tens, pts_tens], axis=-1)

    @staticmethod
    def numpy_feature_types(max_num_points_in_route: int) -> dict[tuple[int, int], str]:
        crr_np_dim = max(r for _, r in Courier.numpy_feature_types().keys())
        return Courier.numpy_feature_types() | {
            (crr_np_dim, crr_np_dim + 2): 'numbers',
            (crr_np_dim + 2, crr_np_dim + 2 + 2 * max_num_points_in_route): 'coords',
        }


@dataclass
class Gamble:
    couriers: list[Courier]
    claims: list[Claim]
    orders: list[Order]
    dttm_start: datetime
    dttm_end: datetime


@dataclass
class Assignment:
    """
    Pairs of courier ID and claim ID
    """
    ids: list[tuple[int, int]]
