import random
from typing import List, Tuple

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
        
class Courier:
    def __init__(self, position: Point) -> None:
        self.id = random.randint(0, int(1e9))
        self.position = position
        self.busy_untill = 0
    
    def __repr__(self) -> str:
        return 'Courier;  id: ' + str(self.id) + '; pos: ' + self.position.__repr__()

    def plot(self, fig):
        return self.position.plot(fig, 'red', 10, '.')

class Order:
    def __init__(self, point_from: Point, point_to: Point) -> None:
        self.id = random.randint(0, int(1e9))
        self.point_from = point_from
        self.point_to = point_to
    
    def __repr__(self) -> str:
        return 'Order; id: ' + str(self.id) + '; from: ' + self.point_from.__repr__() + '; to: ' + self.point_to.__repr__()

    def plot(self, fig):
        self.point_from.plot(fig, 'green', 5, '^')
        self.point_to.plot(fig, 'blue', 5, 'v')
        return fig

# class Route:
#     def __init__(self, points: List[Point]) -> None:
#         self.id = random.randint(0, int(1e9))
#         self.points = points
#         self.get_distace()

#     def get_distace(self):
#         self.distance = 0
#         p_from = self.points[0]
#         for p in self.points[1:]:
#             self.distance += distance(p, p_from)
#             p_from = p

#     def start(self):
#         return self.points[0]

#     def end(self):
#         return self.points[-1]
        
class ActiveRoute:
    def __init__(self, courier: Courier, order: Order) -> None:
        self.courier = courier
        self.order = order
        self.target_point = order.point_from
        self.eta_period = True
        self.is_active = True

    def next(self, step):
        if not self.is_active:
            return
        if distance(self.courier.position, self.target_point) < step:
            self.courier.position = self.target_point
            if self.eta_period:
                self.eta_period = False
                self.target_point = self.order.point_to
            else:
                self.is_active = False
        else:
            self.courier.position += step * (self.target_point - self.courier.position).normalize()
            
        return self

    def __repr__(self) -> str:
        return 'ActiveRoute: [' + self.courier.__repr__() + '; ' + self.order.__repr__() + ']'

    def plot(self, fig):
        self.courier.plot(fig)
        self.order.plot(fig)
        return fig

def distance(first_point: Point, second_point: Point):
    return ((first_point.x - second_point.x)**2 + (first_point.y - second_point.y)**2)**0.5
    
# def generate_route_from_list_of_orders(orders: List[Order]):
#     assert len(orders) == 1
#     return Route([orders[0].point_from, orders[0].point_to])

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
