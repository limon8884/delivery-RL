import random
import numpy as np

from .objects import (
    Point,
    Gamble,
    Courier,
    Order,
    Claim,
)


def get_random_point(corner_bounds: tuple[Point, Point] = (Point(0.0, 0.0), Point(1.0, 1.0))) -> Point:
    left_lower_bound, right_upper_bound = corner_bounds
    x = random.random() * (right_upper_bound.x - left_lower_bound.x) + left_lower_bound.x
    y = random.random() * (right_upper_bound.y - left_lower_bound.y) + left_lower_bound.y

    return Point(x, y)


def write_in_txt_file(path: str, content: str) -> None:
    with open(path, 'a') as f:
        f.write(content)
        f.write('\n')


def compulte_claims_to_couriers_distances(gamble: Gamble, distance_norm_constant: float) -> np.ndarray:
    result = []
    for claim in gamble.claims:
        result.append([])
        for courier in gamble.couriers:
            dist = Point.distance(claim.source_point, courier.position) / distance_norm_constant
            result[-1].append(dist)
        for order in gamble.orders:
            dist = Point.distance(claim.source_point, order.courier.position) / distance_norm_constant
            result[-1].append(dist)
        result[-1].append(-1)
    return np.array(result)
