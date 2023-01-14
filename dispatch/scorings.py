from typing import List
import numpy as np

from dispatch.dispatch import Edge
from utils import *

class ETAScoring:
    def __init__(self) -> None:
        pass
        # self.max_distance_to_point_A = max_distance_to_point_A

    def __call__(self, orders: List[Order], couriers: List[Courier]) -> np.ndarray:
        edges = []
        for order in orders:
            edges.append([])
            for courier in couriers:
                dist = distance(courier.position, order.point_from)
                # if dist > self.max_distance_to_point_A:
                #     dist = 1000
                edges[-1].append(dist)

        return -np.array(edges)


