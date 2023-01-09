from typing import List
from dispatch import Edge
import numpy as np

from utils import *

class ETAScoring:
    def __init__(self) -> None:
        pass

    def __call__(self, orders: Order, couriers: Courier) -> np.ndarray:
        edges = []
        for order in orders:
            edges.append([])
            for courier in couriers:
                dist = distance(courier.position, order.point_from)
                if dist > self.max_distance_to_point_A:
                    dist = np.Inf
                edges[-1].append(dist)

        return -np.array(edges)