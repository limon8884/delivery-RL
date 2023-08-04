import random
from typing import Tuple


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __rmul__(self, other):
        return Point(self.x * other, self.y * other)

    def __rtruediv__(self, other):
        return Point(self.x / other, self.y / other)

    def normalize(self):
        n = (self.x**2 + self.y**2)**0.5
        if n == 0:
            return Point(self.x, self.y)
        return Point(self.x / n, self.y / n)

    def __repr__(self) -> str:
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

    def plot(self, fig, color='black', size=10, marker='.'):
        fig.scatter(self.x, self.y, color=color, s=size, marker=marker)
        return fig


def get_random_point(corner_bounds: Tuple[Point, Point] = (Point(0, 0), Point(10, 10))):
    left_lower_bound, right_upper_bound = corner_bounds
    x = random.random() * (right_upper_bound.x - left_lower_bound.x) + left_lower_bound.x
    y = random.random() * (right_upper_bound.y - left_lower_bound.y) + left_lower_bound.y

    return Point(x, y)
