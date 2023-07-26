from typing import Tuple
from src.objects.point import Point
from src.objects.courier import Courier, random_courier
from src.objects.order import Order, random_order
from src.objects.utils import distance


class ActiveRoute:
    def __init__(self, courier: Courier, order: Order, creation_time: int, id: int = None) -> None:
        if id is not None:
            self.id = id
        self.courier = courier
        self.order = order
        self.target_point = order.point_from
        self.eta_period = True
        self.is_active = True

        self.order.FoundCourier(creation_time)

    def next(self, step, time):
        if not self.is_active:
            return
        self.courier.next(time)
        self.order.next(time)
        if distance(self.courier.position, self.target_point) < step:
            self.courier.position = self.target_point
            if self.eta_period:
                self.order.SourcePointVisited(time)
                self.eta_period = False
                self.target_point = self.order.point_to
            else:
                self.is_active = False
                self.order.CompleteOrder(time)
        else:
            self.courier.position += step * (self.target_point - self.courier.position).normalize()

        return self

    def __repr__(self) -> str:
        return 'ActiveRoute: [' + self.courier.__repr__() + '; ' + self.order.__repr__() + ']'

    def plot(self, fig):
        self.courier.plot(fig)
        self.order.plot(fig)
        return fig


def random_ar(corner_bounds: Tuple[Point], id=None):
    return ActiveRoute(random_courier(corner_bounds), random_order(corner_bounds), 0, id)
