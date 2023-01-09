from typing import List, Tuple
import itertools
import numpy as np

from utils import *
from dispatch.utils import *
from dispatch.generators import FullGenerator
from dispatch.scorings import ETAScoring
from dispatch.solvers import HungarianSolver

class Dispatch:
    '''
    Gets 3 lists as input:
    -free orders
    -free couriers
    -active routes
    returns pairs of indexes (order, courier) of first 2 lists - assigments
    '''
    def __init__(self, max_distance_to_point_A=2.5) -> None:
        self.scoring = ETAScoring(max_distance_to_point_A)
        self.solver = HungarianSolver()

        self.statistics = {
            "avg_score": [],
            "num_assignments": []
        }

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

        self.statistics['avg_scores'].append(np.mean([scores[ass[0], ass[1]] for ass in assignments]))
        self.statistics['num_assignments'].append(len(assignments))

        return assignments
