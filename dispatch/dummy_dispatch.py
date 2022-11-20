from typing import List, Tuple

from utils import *

class DummyDispatch:
    def __init__(self) -> None:
        pass

    def __call__(self, 
        busy_couriers_with_complite_times: List[Tuple[float, Courier]], 
        free_couriers: List[Courier],
        routes: List[Route]
    ) -> List[Tuple[Courier, Route]]:
        assignments = []

        min_num = min(len(free_couriers), len(routes))
        for i in range(min_num):
            assignments.append((free_couriers[i], routes[i]))
        
        assert self.check_no_collisions(assignments)
        return assignments
    
    def check_no_collisions(self, propositions: List[Tuple[Courier, Route]]):
        courier_set = set()
        route_set = set()
        for courier, route in propositions:
            if courier.id in courier_set or route.id in route_set:
                return False
            courier_set.add(courier.id)
            route_set.add(route.id)
        return True