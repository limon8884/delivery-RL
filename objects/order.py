import random
from typing import Tuple
from objects.point import Point, get_random_point


class Order:
    def __init__(self,
                 point_from: Point,
                 point_to: Point,
                 creation_time: int,
                 expire_time: int,
                 id: int = None
                 ) -> None:
        if id is None:
            self.id = random.randint(0, int(1e9))
        else:
            self.id = id

        self.point_from = point_from
        self.point_to = point_to

        self.creation_time = creation_time
        self.expire_time = expire_time
        self.is_completed = False

    def FoundCourier(self, time):
        self.courier_found_time = time

    def SourcePointVisited(self, time):
        self.source_point_visited_time = time

    def CompleteOrder(self, time):
        self.is_completed = True
        self.complete_time = time

    def next(self, time) -> None:
        pass

    def __repr__(self) -> str:
        return ('Order; id: '
                + str(self.id)
                + '; from: '
                + self.point_from.__repr__()
                + '; to: '
                + self.point_to.__repr__()
                )

    def plot(self, fig):
        self.point_from.plot(fig, 'green', 5, '^')
        self.point_to.plot(fig, 'blue', 5, 'v')
        return fig


def random_order(corner_bounds: Tuple[Point], id=None):
    return Order(get_random_point(corner_bounds), get_random_point(corner_bounds), 0, 10, id)
