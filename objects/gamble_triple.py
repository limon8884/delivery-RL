import random
from typing import List, Tuple

from objects.point import Point, get_random_point
from objects.order import Order, random_order
from objects.courier import Courier, random_courier
from objects.active_route import ActiveRoute, random_ar

class GambleTriple:
    def __init__(self, orders: List[Order], couriers: List[Courier], active_routes: List[ActiveRoute]) -> None:
        self.orders = orders
        self.couriers = couriers
        self.active_routes = active_routes

    def __repr__(self) -> str:
        return 'Orders:\n' + '\n'.join([o.__repr__() for o in self.orders]) \
            + '\nCouriers:\n' + '\n'.join([c.__repr__() for c in self.couriers]) \

    def plot(self, fig):
        for i, order in enumerate(self.orders):
            order.plot(fig)
            fig.annotate(str(i) + 's', (order.point_from.x, order.point_from.y))
            fig.annotate(str(i) + 'f', (order.point_to.x, order.point_to.y))
        for i, courier in enumerate(self.couriers):
            courier.plot(fig)
            fig.annotate(str(i), (courier.position.x, courier.position.y))


def random_triple(corner_bounds: Tuple[Point], max_items=10, same_number=False):
    num_orders = int(random.random() * max_items) + 1
    num_couriers = int(random.random() * max_items) + 1
    num_ars = int(random.random() * max_items) + 1
    if same_number:
        num_couriers = num_orders
        num_ars = num_orders
    os = [random_order(corner_bounds) for _ in range(num_orders)]
    cs = [random_courier(corner_bounds) for _ in range(num_couriers)]
    ars = [random_ar(corner_bounds) for _ in range(num_ars)]
    
    return GambleTriple(os, cs, ars)