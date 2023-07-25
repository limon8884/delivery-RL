from scipy.optimize import linear_sum_assignment
import numpy as np


class HungarianSolver:
    def __init__(self) -> None:
        pass

    def __call__(self, scores: np.ndarray):
        return linear_sum_assignment(scores, maximize=True)
