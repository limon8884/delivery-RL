import random
from typing import List, Tuple

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        
class Courier:
    def __init__(self, position: Point) -> None:
        self.id = random.randint(int(1e9))
        self.position = position
        self.busy_untill = 0

class Order:
    def __init__(self, point_from: Point, point_to: Point) -> None:
        self.id = random.randint(int(1e9))
        self.point_from = point_from
        self.point_to = point_to

class Route:
    def __init__(self, points: List[Point]) -> None:
        self.id = random.randint(int(1e9))
        self.points = points
        self.get_distace()

    def get_distace(self):
        self.distance = 0
        p_from = self.points[0]
        for p in self.points[1:]:
            self.distance += distance(p, p_from)
            p_from = p

    def start(self):
        return self.points[0]

    def end(self):
        return self.points[-1]
        

def distance(first_point: Point, second_point: Point):
    return ((first_point.x - second_point.x)**2 + (second_point.y + second_point.y)**2)**0.5
    
def generate_route_from_list_of_orders(orders: List[Order]):
    assert len(orders) == 1
    return Route([orders[0].point_from, orders[0].point_to])

def get_random_point(corner_bounds: Tuple[Point]):
    left_lower_bound, right_upper_bound = corner_bounds
    x = random.random() * (right_upper_bound.x - left_lower_bound.x) + left_lower_bound.x
    y = random.random() * (right_upper_bound.y - left_lower_bound.y) + left_lower_bound.y

    return Point(x, y)

def get_random_courier(corner_bounds: Tuple[Point]):
    return Courier(get_random_point(corner_bounds))

def get_random_order(corner_bounds: Tuple[Point]):
    return Order(get_random_point(corner_bounds), get_random_point(corner_bounds))

def distance_to_time(distance):
    return distance
