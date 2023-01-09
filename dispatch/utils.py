from utils import *

class Edge:
    def __init__(self, order: Order, courier: Courier) -> None:
        self.order = order
        self.courier = courier
        self.score = None