from typing import List, Tuple
from utils import *
import heapq
import random

class BaseSimulator:
    def __init__(self,
        dispatch, 
        num_couriers=10, 
        num_orders_every_gamble=2, 
        corner_bounds: Tuple[Point] = (Point(0, 0), Point(1, 1))
    ) -> None:
        self.num_couriers = num_couriers
        self.num_orders_every_gamble = num_orders_every_gamble
        self.corner_bounds = corner_bounds
        self.dispatch = dispatch

        self.gamble_iteration = 0

        self.free_orders = []
        self.free_couriers = []
        self.active_routes = []

        self.finished_couriers = []
        self.finished_orders = []

    def Update(self):
        self.UpdateOrders()
        self.UpdateCouriers()
        self.UpdateActiveRoutes()

    def Assign(self):
        assignments = self.dispatch()

    def UpdateOrders(self):
        pass

    def UpdateCouriers(self):
        pass

    def UpdateActiveRoutes(self):
        pass

    def Free():
        pass
        

    def gamble_iteration_to_time(self):
        return float(self.gamble_iteration)

    def update_free_orders(self):
        self.unassigned_orders += [get_random_order(self.corner_bounds) for _ in range(self.num_orders_every_gamble)]

    def get_propositions(self):
        return self.dispatch(self.busy_couriers_with_time_of_complite, self.free_couriers, self.unassigned_orders)

    def assign_couriers(self, propositions: List[Tuple[Courier, Route]]):
        for courier, route in propositions:
            route_complite_time = self.gamble_iteration_to_time() + distance_to_time(route.distance)
            heapq.heappush(self.busy_couriers_with_time_of_complite, (route_complite_time, courier))
    
    def unassign_couriers(self):
        while self.busy_couriers_with_time_of_complite:
            complite_time, _ = self.busy_couriers_with_time_of_complite[0]
            if complite_time > self.gamble_iteration_to_time():
                break
            _, courier = heapq.heappop(self.busy_couriers_with_time_of_complite)
            self.free_couriers.append(courier)        

    def next(self):
        self.gamble_iteration += 1
        self.unassign_couriers()
        propositions = self.get_propositions()
        self.assign_couriers(propositions)


