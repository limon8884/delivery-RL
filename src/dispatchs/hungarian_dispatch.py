from scipy.optimize import linear_sum_assignment

from src.dispatchs.base_dispatch import BaseDispatch
from src.dispatchs.scorers import BaseScorer
from src.objects import (
    Gamble,
    Assignment,
)


class HungarianDispatch(BaseDispatch):
    """
    Solves the problem of maximum weight matching in bipaired graph
    """
    def __init__(self, scorer: BaseScorer) -> None:
        self.scorer = scorer

    def __call__(self, gamble: Gamble) -> Assignment:
        fake_clm_idx = len(gamble.claims)
        if fake_clm_idx == 0:
            return Assignment([])
        scores = self.scorer.score(gamble)
        crr_idxs, clm_idxs = linear_sum_assignment(scores, maximize=True)
        return Assignment([
            (gamble.couriers[crr_idx].id, gamble.claims[clm_idx].id)
            for crr_idx, clm_idx in zip(crr_idxs, clm_idxs)
            if clm_idx != fake_clm_idx
        ])
