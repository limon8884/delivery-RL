from src.dispatchs.base_dispatch import BaseDispatch
from src.objects import (
    Gamble,
    Assignment,
)


class RandomDispatch(BaseDispatch):
    """
    Iterate through the couriers and assign them randomly
    """
    def __call__(self, gamble: Gamble) -> Assignment:
        if len(gamble.claims) == 0:
            return Assignment([])
        couriers = {crr.id for crr in gamble.couriers}
        claims = {clm.id for clm in gamble.claims}
        assignments: list[tuple[int, int]] = []

        while len(couriers) > 0 and len(claims) > 0:
            clm_id = next(iter(claims))
            crr_id = next(iter(couriers))
            assignments.append((crr_id, clm_id))
            claims.remove(clm_id)
            couriers.remove(crr_id)

        return Assignment(assignments)
