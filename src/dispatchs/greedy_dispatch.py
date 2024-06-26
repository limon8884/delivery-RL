import numpy as np

from src.dispatchs.base_dispatch import BaseDispatch
from src.dispatchs.scorers import BaseScorer
from src.objects import (
    Gamble,
    Assignment,
)


class GreedyDispatch(BaseDispatch):
    """
    Iterate through the couriers and assign the best options in greedy manner
    """
    def __init__(self, scorer: BaseScorer) -> None:
        self.scorer = scorer

    def __call__(self, gamble: Gamble) -> Assignment:
        if len(gamble.claims) == 0:
            return Assignment([])
        scores = self.scorer.score(gamble)
        assert scores.shape == (len(gamble.couriers), len(gamble.claims) + 1)
        assigned_claim_idxs: set[int] = set()
        assignments: list[tuple[int, int]] = []
        fake_clm_idx = len(gamble.claims)

        for crr_idx, clm_idx in enumerate(np.argmax(scores, axis=-1)):
            if clm_idx not in assigned_claim_idxs and assigned_claim_idxs != fake_clm_idx:
                assignments.append((gamble.couriers[crr_idx].id, gamble.claims[clm_idx].id))
                assigned_claim_idxs.add(clm_idx)

        return Assignment(assignments)


class GreedyDispatch2(BaseDispatch):
    """
    Iterate through the couriers and assign the best options in greedy manner
    """
    def __init__(self, scorer: BaseScorer) -> None:
        self.scorer = scorer

    def __call__(self, gamble: Gamble) -> Assignment:
        if len(gamble.claims) == 0:
            return Assignment([])
        scores = self.scorer.score(gamble)
        assert scores.shape == (len(gamble.couriers), len(gamble.claims) + 1)
        couriers_assigned = np.zeros(len(gamble.couriers), dtype=np.int32)
        assignments: list[tuple[int, int]] = []

        for clm_idx in range(len(gamble.claims)):
            if (couriers_assigned == 1).all():
                break
            dists = scores[:, clm_idx] - couriers_assigned
            crr_idx = np.argmax(dists)
            assignments.append((gamble.couriers[crr_idx].id, gamble.claims[clm_idx].id))
            couriers_assigned[crr_idx] = 1

        return Assignment(assignments)
