import typing as tp
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src_new.database.database import Database, TableName, Event


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


@dataclass
class Route:
    points: list[Point]
    claim_ids: list[int]
    point_types: list['PointType']

    class PointType(Enum):
        SOURCE = 1
        DESTINATION = 2

    def distance(self) -> float:
        if len(self.points) == 0:
            return 0
        total_dist: float = 0
        current_point = self.points[0]
        for next_point in self.points[1:]:
            total_dist += Point.distance(current_point, next_point)
            current_point = next_point
        return total_dist


class Item:
    """
    A base class for Courier, Claim and Order
    """
    def __init__(self, id: int, db: Database | None) -> None:
        """
        id - id of item
        dttm - creation datetime in the system
        """
        self.id: int = id
        self._dttm: tp.Optional[datetime] = None
        self._db: tp.Optional[Database] = db

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


class Courier(Item):
    class Status(Enum):
        FREE = 'free'
        PROCESS = 'process'
        OFFLINE = 'offline'

    def __init__(self,
                 id: int,
                 position: Point,
                 start_dttm: datetime,
                 end_dttm: datetime,
                 courier_type: str,
                 db: Database | None = None,
                 ) -> None:
        super().__init__(id, db)
        self.position = position
        self.start_dttm = start_dttm
        self.end_dttm = end_dttm
        self.courier_type = courier_type

        self.status: Courier.Status = Courier.Status.FREE
        if self._db is not None:
            self._db.insert(TableName.COURIER_TABLE, id, start_dttm, Event.COURIER_STARTED)

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
            if self._db is not None:
                self._db.insert(TableName.COURIER_TABLE, self.id, self._dttm, Event.COURIER_ENDED)

    def done(self) -> bool:
        return self.status is Courier.Status.OFFLINE
        # if self._dttm is None:
        #     return False
        # return self.end_dttm <= self._dttm

    # def done_dttm(self) -> datetime:
    #     assert self.done()
    #     return self._dttm


class Claim(Item):
    class Status(Enum):
        UNASSIGNED = 'unassigned'
        ASSIGNED = 'assigned'
        # IN_ARRIVAL = 'in_arrival'
        # PICKUPED = 'pickuped'
        COMPLETED = 'completed'
        CANCELLED = 'cancelled'

    def __init__(self,
                 id: int,
                 source_point: Point,
                 destination_point: Point,
                 creation_dttm: datetime,
                 cancell_if_not_assigned_dttm: datetime,
                 waiting_on_point_source: timedelta,
                 waiting_on_point_destination: timedelta,
                 db: Database | None = None,
                 ) -> None:
        super().__init__(id, db)
        self.source_point = source_point
        self.destination_point = destination_point
        self.creation_dttm = creation_dttm
        self.cancell_if_not_assigned_dttm = cancell_if_not_assigned_dttm
        self.waiting_on_point_source = waiting_on_point_source
        self.waiting_on_point_destination = waiting_on_point_destination

        self.status: Claim.Status = Claim.Status.UNASSIGNED
        if self._db is not None:
            self._db.insert(TableName.CLAIM_TABLE, id, creation_dttm, Event.CLAIM_CREATED)

    def next(self, current_time: datetime) -> None:
        if self.done():
            raise RuntimeError('Claim next when done')
        super().next(current_time)
        if self.cancell_if_not_assigned_dttm < self._dttm and self.status is Claim.Status.UNASSIGNED:
            self.status = Claim.Status.CANCELLED
            if self._db is not None:
                self._db.insert(TableName.CLAIM_TABLE, self.id, self._dttm, Event.CLAIM_CANCELLED)

    def assign(self):
        assert self.status is Claim.Status.UNASSIGNED
        self.status = Claim.Status.ASSIGNED

    def complete(self, seconds_to_act: int):
        assert self.status is Claim.Status.ASSIGNED
        self.status = Claim.Status.COMPLETED
        if self._db is not None:
            self._db.insert(TableName.CLAIM_TABLE, self.id,
                            self._dttm - timedelta(seconds=seconds_to_act), Event.CLAIM_COMPLETED)

    def done(self) -> bool:
        return self.status in (Claim.Status.COMPLETED, Claim.Status.CANCELLED)
        # if self._dttm is None:
        #     return False
        # return self.cancell_if_not_assigned_dttm < self._dttm

    # def done_dttm(self) -> datetime:
    #     assert self.done()
    #     return self._dttm


class Order(Item):
    class Status(Enum):
        IN_ARRIVAL = 'in_arrival'
        IN_PROCESS = 'in_process'
        COMPLETED = 'completed'

    def __init__(self,
                 id: int,
                 creation_dttm: datetime,
                 courier: Courier,
                 route: Route,
                 claims: list[Claim],
                 db: Database | None = None,
                 ) -> None:
        super().__init__(id, db)
        self.creation_dttm = creation_dttm
        self.courier = courier
        self.claims = {claim.id: claim for claim in claims}
        self.route = route
        # self.wait_on_point_secs = waiting_on_point.total_seconds()

        for claim in claims:
            claim.assign()
        self.status = Order.Status.IN_ARRIVAL
        self.courier.status = Courier.Status.PROCESS
        if self._db is not None:
            self._db.insert(TableName.ORDER_TABLE, id, self.creation_dttm, Event.ORDER_CREATED)

        self._dttm = creation_dttm
        self._next_point_idx = 0
        # self._current_claim_id = self.route.claim_ids[0]
        self._seconds_to_wait = None

    def done(self) -> bool:
        return self.status is Order.Status.COMPLETED

    # def done_dttm(self) -> datetime:
    #     assert self.done()
    #     return self._dttm

    def get_arrival_distance(self) -> float:
        assert self.status is Order.Status.IN_ARRIVAL, 'courier is not in arrival'
        return Point.distance(self.courier.position, self.route.points[0])

    def get_rest_distance(self) -> float:
        if self._next_point_idx != len(self.route.points):
            return Point.distance(self.courier.position, self.route.points[self._next_point_idx]) \
                + Route(self.route.points[self._next_point_idx:], [], []).distance()
        return 0

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
            if self._next_point_idx == len(self.route.points):
                self._finish_order(seconds_to_act)
                return
            seconds_to_act = self._move_courier(seconds_to_act, speed)
            seconds_to_act = self._wait_on_point_func(seconds_to_act)

    def _finish_order(self, seconds_to_act: int) -> None:
        self.status = Order.Status.COMPLETED
        self.courier.status = Courier.Status.OFFLINE if self.courier.is_time_off() else Courier.Status.FREE
        if self._db is not None:
            self._db.insert(TableName.ORDER_TABLE, self.id,
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
        if self.route.point_types[self._next_point_idx] is Route.PointType.DESTINATION:
            self.claims[self.route.claim_ids[self._next_point_idx]].complete(seconds_to_act)
        self._next_point_idx += 1
        return seconds_to_act

    def _move_courier(self, seconds_to_act: int, speed: float) -> int:
        '''
        Returns remeining seconds to act
        '''
        if self._seconds_to_wait is not None:
            return seconds_to_act
        next_point = self.route.points[self._next_point_idx]
        dist_to_next_point = Point.distance(self.courier.position, next_point)
        if dist_to_next_point > speed * seconds_to_act:
            new_courier_pos = self.courier.position + \
                (next_point - self.courier.position).normalize() * speed * seconds_to_act
            self.courier.set_position(new_courier_pos)
            return 0
        seconds_to_act -= dist_to_next_point / speed
        self._seconds_to_wait = \
            self.claims[self.route.claim_ids[self._next_point_idx]].waiting_on_point_source.total_seconds() \
            if self.route.point_types[self._next_point_idx] is Route.PointType.SOURCE \
            else self.claims[self.route.claim_ids[self._next_point_idx]].waiting_on_point_destination.total_seconds()
        self.courier.set_position(next_point)
        self.status = Order.Status.IN_PROCESS
        return seconds_to_act


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
