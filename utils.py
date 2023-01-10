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
    def __init__(self, position: Point, creation_time: int, off_time: int) -> None:
        self.id = random.randint(0, int(1e9))
        self.position = position
        self.creation_time = creation_time
        self.off_time = off_time
        self.actual_off_time = None

    def OffCourier(self, time):
        self.actual_off_time = time

    def next(self, time) -> None:
        pass

    def __repr__(self) -> str:
        return 'Courier;  id: ' + str(self.id) + '; pos: ' + self.position.__repr__()

    def plot(self, fig):
        return self.position.plot(fig, 'red', 10, '.')

class Order:
    def __init__(self, point_from: Point, point_to: Point, creation_time: int, expire_time: int) -> None:
        self.id = random.randint(0, int(1e9))

        self.point_from = point_from
        self.point_to = point_to

        self.creation_time = creation_time
        self.expire_time = expire_time
        self.is_completed = False

    def FoundCourier(self, time):
        self.courier_found_time = time

    def SourcePointVisited(self, time):
        self.source_point_visited_time = time

    def CompleteOrder(self, time):
        self.is_completed = True
        self.complete_time = time

    def next(self, time) -> None:
        pass
    
    def __repr__(self) -> str:
        return 'Order; id: ' + str(self.id) + '; from: ' + self.point_from.__repr__() + '; to: ' + self.point_to.__repr__()

    def plot(self, fig):
        self.point_from.plot(fig, 'green', 5, '^')
        self.point_to.plot(fig, 'blue', 5, 'v')
        return fig
       
class ActiveRoute:
    def __init__(self, courier: Courier, order: Order, creation_time: int) -> None:
        self.courier = courier
        self.order = order
        self.target_point = order.point_from
        self.eta_period = True
        self.is_active = True

        self.order.FoundCourier(creation_time)

    def next(self, step, time):
        if not self.is_active:
            return
        self.courier.next(time)
        self.order.next(time)
        if distance(self.courier.position, self.target_point) < step:
            self.courier.position = self.target_point
            if self.eta_period:
                self.order.SourcePointVisited(time)
                self.eta_period = False
                self.target_point = self.order.point_to
            else:
                self.is_active = False
                self.order.CompleteOrder(time)
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
    
def get_random_point(corner_bounds: Tuple[Point]):
    left_lower_bound, right_upper_bound = corner_bounds
    x = random.random() * (right_upper_bound.x - left_lower_bound.x) + left_lower_bound.x
    y = random.random() * (right_upper_bound.y - left_lower_bound.y) + left_lower_bound.y

    return Point(x, y)
