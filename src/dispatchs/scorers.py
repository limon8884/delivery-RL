import numpy as np
from sklearn.metrics import pairwise_distances

from src.objects import (
    Gamble,
)


class BaseScorer:
    """
    Builds a weighted bipaired graph with non-negative weights
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
        assert np.all(weights >= 0)
        return weights

    def score_batch(self, gambles: list[Gamble]) -> np.ndarray:
        raise NotImplementedError
