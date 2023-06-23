from typing import Any, List, Tuple, Dict, Sequence
from utils import *
import numpy as np
import heapq
import random
import json

from objects.point import Point, get_random_point
from objects.order import Order
from objects.courier import Courier
from objects.active_route import ActiveRoute
from objects.gamble_triple import GambleTriple
from objects.utils import *

from simulator.timer import Timer

class _Generator:
    def __init__(self, corner_bounds, timer: Timer):
        self.timer = timer
        self.corner_bounds = corner_bounds
        self.current_id = 0

    def __call__(self):
        self.current_id += 1

    def reset(self):
        self.current_id = 0

class CourierGenerator(_Generator):
    def __init__(self, corner_bounds, timer: Timer, courier_live_time: int):
        super().__init__(corner_bounds, timer)
        self.courier_live_time = courier_live_time

    def __call__(self):
        courier = Courier(
                get_random_point(self.corner_bounds),
                self.timer(),
                self.timer() + self.courier_live_time,
                self.current_id
        )
        super().__call__()
        return courier

class OrderGenerator(_Generator):
    def __init__(self, corner_bounds, timer: Timer, order_live_time: int):
        super().__init__(corner_bounds, timer)
        self.order_live_time = order_live_time

    def __call__(self):
        order = Order(
                get_random_point(self.corner_bounds),
                get_random_point(self.corner_bounds),
                self.timer(),
                self.timer() + self.order_live_time,
                self.current_id
        )
        super().__call__()
        return order

class ActiveRouteGenerator(_Generator):
    def __init__(self, corner_bounds, timer: Timer):
        super().__init__(corner_bounds, timer)

    def __call__(self, order: Order, courier: Courier):
        active_route = ActiveRoute(
                courier,
                order,
                self.timer(),
                self.current_id
        )
        super().__call__()
        return active_route

