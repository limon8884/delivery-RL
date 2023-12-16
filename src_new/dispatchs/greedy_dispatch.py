import numpy as np

from src_new.dispatchs.base_dispatch import BaseDispatch
from src_new.dispatchs.scorers import BaseScorer
from src_new.objects import (
    Gamble,
    Assignment,
)


class GreedyDispatch(BaseDispatch):
    """
    Iterate through the couriers and assign the best options in greedy manner
    """
    def __init__(self, scorer: BaseScorer) -> None:
        super().__init__(scorer)

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
