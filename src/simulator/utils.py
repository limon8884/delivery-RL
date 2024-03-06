from datetime import datetime
from dataclasses import dataclass

from src.objects import (
    Claim,
    Courier,
)


@dataclass
class CityStamp:
    """
    A Snapshot of a city for a given time interval.
    Contains new couriers on a line, new claims and context
    """
    from_dttm: datetime
    to_dttm: datetime
    couriers: list[Courier]
    claims: list[Claim]
