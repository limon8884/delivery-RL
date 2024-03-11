import random

from .objects import (
    Point
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
