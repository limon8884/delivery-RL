import random
from typing import Tuple
from objects.point import Point, get_random_point


class Courier:
    def __init__(self, position: Point, creation_time: int, off_time: int, id: int = None) -> None:
        if id is None:
            self.id = id or random.randint(0, int(1e9))
        else:
            self.id = id

        self.position = position
        self.creation_time = creation_time
        self.off_time = off_time
        self.actual_off_time = None

    def OffCourier(self, time):
        self.actual_off_time = time

    def next(self, time) -> None:
        pass

    def __repr__(self) -> str:
        return 'Courier;  id: ' + str(self.id) + '; pos: ' + self.position.__repr__()

    def plot(self, fig):
        return self.position.plot(fig, 'red', 10, '.')


def random_courier(corner_bounds: Tuple[Point], id=None):
    return Courier(get_random_point(corner_bounds), 0, 10, id)
