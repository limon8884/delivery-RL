from typing import List, Dict
from utils import *

class Edge:
    def __init__(self, order: Order, courier: Courier) -> None:
        self.order = order
        self.courier = courier
        self.score = None

class GambleTriple:
    def __init__(self, orders: List[Order], couriers: List[Courier], active_routes: List[ActiveRoute]) -> None:
        self.orders = orders
        self.couriers = couriers
        self.active_routes = active_routes

def random_triple(corner_bounds: Tuple[Point], max_items=10):
    os = [random_order(corner_bounds) for _ in range(int(random.random() * max_items) + 1)]
    cs = [random_courier(corner_bounds) for _ in range(int(random.random() * max_items) + 1)]
    ars = [random_ar(corner_bounds) for _ in range(int(random.random() * max_items) + 1)]
    
    return GambleTriple(os, cs, ars)