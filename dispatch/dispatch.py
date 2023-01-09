from typing import List, Tuple
import itertools
import numpy as np

from utils import *
from dispatch.utils import *
from generators import FullGenerator
from scorings import ETAScoring
from solvers import HungarianSolver

class Dispatch:
    '''
    Gets 3 lists as input:
    -free orders
    -free couriers
    -active routes
    returns pairs of indexes (order, courier) of first 2 lists - assigments
    '''
    def __init__(self) -> None:
        self.generator = FullGenerator
        self.scoring = ETAScoring
        self.solver = HungarianSolver
        self.max_distance_to_point_A = 2.5

    def __call__(self, 
        free_orders: List[Order], 
        free_couriers: List[Courier], 
        active_routes: List[ActiveRoute]
    ) -> List[Tuple[int, int]]:
        scores = self.scoring(free_orders, free_couriers)        
        assigned_order_idxs, assigned_courier_idxs = self.solver(scores)
        assignments = []
        for o_idx, c_idx in zip(assigned_order_idxs, assigned_courier_idxs):
            if scores[o_idx][c_idx] != -np.Inf:
                assignments.append((o_idx, c_idx))

        return assignments
