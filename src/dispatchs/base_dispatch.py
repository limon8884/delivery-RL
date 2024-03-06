# from src.dispatchs.scorers import BaseScorer
from src.objects import (
    Gamble,
    Assignment,
)


class BaseDispatch:
    """
    A base class for dipatch.
    Dispatch is a black box which assignes couriers to claims using scoring
    """
    def __call__(self, gamble: Gamble) -> Assignment:
        """
        Makes assignments
        """
        raise NotImplementedError
