from src_new.dispatchs.scorers import BaseScorer
from src_new.objects import (
    Gamble,
    Assignment,
)


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
