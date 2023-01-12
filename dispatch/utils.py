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