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