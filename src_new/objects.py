from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


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
    def __init__(self, id: int) -> None:
        """
        id - id of item
        dttm - creation datetime in the system
        """
        self.id = id

    def next(self, current_time: datetime) -> None:
        """
        Shifts the time of item forward
        """
        self._dttm = current_time

    def done(self) -> bool:
        raise NotImplementedError


class Courier(Item):
    class Status(Enum):
        ONLINE = 'online'
        OFFLINE = 'offline'

    def __init__(self,
                 id: int,
                 position: Point,
                 start_dttm: datetime,
                 end_dttm: datetime,
                 courier_type: str,
                 ) -> None:
        super().__init__(id)
        self.position = position
        self.start_dttm = start_dttm
        self.end_dttm = end_dttm
        self.courier_type = courier_type

    def set_position(self, position: Point) -> None:
        self.position = position

    def done(self) -> bool:
        return self.end_dttm <= self._dttm


class Claim(Item):
    class Status(Enum):
        UNASSIGNED = 'unassigned'
        IN_ETA = 'in_eta'
        PICKUPED = 'pickuped'
        COMPLETED = 'completed'
        CANCELLED = 'cancelled'

    def __init__(self,
                 id: int,
                 source_point: Point,
                 destination_point: Point,
                 creation_dttm: datetime,
                 cancell_if_not_assigned_dttm: datetime,
                 waiting_on_point: timedelta
                 ) -> None:
        super().__init__(id)
        self.source_point = source_point
        self.destination_point = destination_point
        self.creation_dttm = creation_dttm
        self.cancell_if_not_assigned_dttm = cancell_if_not_assigned_dttm
        self.waiting_on_point = waiting_on_point

    def done(self) -> bool:
        return self.cancell_if_not_assigned_dttm < self._dttm


class Order(Item):
    class Status(Enum):
        IN_ETA = 'in_eta'
        IN_PROCESS = 'in_process'
        COMPLETED = 'completed'

    def __init__(self,
                 id: int,
                 creation_dttm: datetime,
                 courier: Courier,
                 route: Route,
                 waiting_on_point: timedelta,
                 claims: list[Claim],
                 ) -> None:
        super().__init__(id)
        self.creation_dttm = creation_dttm
        self.courier = courier
        self.route = route
        self.wait_on_point_secs = waiting_on_point.total_seconds()
        self.claims = claims
        self.status = Order.Status.IN_ETA

        self._dttm = creation_dttm
        self._next_point_idx = 0
        self._seconds_to_wait = 0

    def done(self) -> bool:
        return self.status is Order.Status.COMPLETED

    def next(self, current_time: datetime, speed: float) -> None:
        seconds_to_act = (current_time - self._dttm).total_seconds()
        super().next(current_time)
        self.courier.next(current_time)
        for claim in self.claims:
            claim.next(current_time)
        if self.status is Order.Status.COMPLETED:
            return
        while seconds_to_act > 0:
            if self._seconds_to_wait == 0 and self._next_point_idx == len(self.route.points):
                self.status = Order.Status.COMPLETED
                return
            if self._seconds_to_wait > 0:
                if self._seconds_to_wait >= seconds_to_act:
                    self._seconds_to_wait -= seconds_to_act
                    return
                else:
                    seconds_to_act -= self._seconds_to_wait
                    self._seconds_to_wait = 0
                    if self._next_point_idx == len(self.route.points):
                        self.status = Order.Status.COMPLETED
                        return
            if self._seconds_to_wait > 0 or self._next_point_idx == len(self.route.points):
                continue
            next_point = self.route.points[self._next_point_idx]
            dist_to_next_point = Point.distance(self.courier.position, next_point)
            if dist_to_next_point > speed * seconds_to_act:
                new_courier_pos = self.courier.position + \
                    (next_point - self.courier.position).normalize() * speed * seconds_to_act
                self.courier.set_position(new_courier_pos)
                return
            else:
                seconds_to_act -= dist_to_next_point / speed
                self._seconds_to_wait = self.wait_on_point_secs
                self.courier.set_position(next_point)
                self._next_point_idx += 1
                self.status = Order.Status.IN_ETA


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


@dataclass
class CityStamp:
    """
    A Snapshot of a city for a given time interval.
    Contains new couriers on a line, new claims and context
    """
    from_dttm: datetime
    to_dttm: datetime
    couriers: list[Courier]
    claims: list[Claim]
