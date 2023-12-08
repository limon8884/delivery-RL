import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from src_new.objects import (
    Gamble,
    Assignment,
)


class BaseScorer:
    """
    Builds a weighted bipaired graph.
    Returnes a matrix of shape (n_couriers, n_claims + 1)
    The last +1 'fake' claim means that we don't want to assign the courier at any claim
    Higher weight of edge - better is the assignment
    """
    def score(self, gamble: Gamble) -> np.ndarray:
        """
        Returns the weight matrix for gamble
        """
        raise NotImplementedError

    def score_batch(self, gambles: list[Gamble]) -> np.ndarray:
        """
        Returns the weight matrix for gamble in batch manner
        """
        scores = [BaseScorer.score(gamble) for gamble in gambles]
        return np.stack(scores, axis=0)


class DistanceScorer(BaseScorer):
    """
    Compute weights between given courier and claim as distance between this courier and
    source point of claim. Then inverse weights with function exp(-x)
    fake claim dimention columns is set to zero
    """
    def score(self, gamble: Gamble) -> np.ndarray:
        if len(gamble.couriers) == 0:
            return np.array([]).reshape((-1, len(gamble.claims) + 1))
        couriers_coords = [[crr.position.x, crr.position.y] for crr in gamble.couriers]
        claims_coords = [[clm.source_point.x, clm.source_point.y] for clm in gamble.claims] + [[0., 0.]]
        distances = pairwise_distances(couriers_coords, claims_coords)
        weights = np.exp(-distances)
        weights[:, -1] = 0.0
        return weights

    def score_batch(self, gambles: list[Gamble]) -> np.ndarray:
        raise NotImplementedError


class BaseDispatch:
    """
    A base class for dipatch.
    Dispatch is a black box which assignes couriers to claims using scoring
    """
    def __init__(self, scorer: BaseScorer) -> None:
        self.scorer = scorer

    def __call__(self, gamble: Gamble) -> Assignment:
        """
        Makes assignments
        """
        raise NotImplementedError


class HungarianDispatch(BaseDispatch):
    """
    Solves the problem of maximum weight matching in bipaired graph
    """
    def __init__(self, scorer: BaseScorer) -> None:
        super().__init__(scorer)

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
