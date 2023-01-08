from typing import List, Tuple, Dict
from utils import *
import heapq
import random
import json

class BaseSimulator:
    def __init__(self,
        dispatch, 
        # num_couriers=10, 
        # num_orders_every_gamble=2, 
        # corner_bounds: Tuple[Point] = (Point(0, 0), Point(1, 1))
    ) -> None:
        self.env_config = None
        # self.num_couriers = num_couriers
        # self.num_orders_every_gamble = num_orders_every_gamble
        # self.corner_bounds = corner_bounds
        self.dispatch = dispatch

        self.gamble_iteration = 0

        self.free_orders = []
        self.free_couriers = []
        self.active_routes = []

        self.finished_couriers = []
        self.finished_orders = []

    def SetParams(self):
        with open('environment_config.json') as f:
            self.env_config = json.load(f)

    def Update(self):
        self.UpdateOrders()
        self.UpdateCouriers()
        self.UpdateActiveRoutes()

    def Assign(self):
        assignments = self.dispatch(self.free_orders, self.free_couriers, self.active_routes)
        for o_idx, c_idx in assignments:
            o = self.free_orders.pop(o_idx)
            c = self.free_couriers.pop(c_idx)
            self.active_routes.append(ActiveRoute(c, o))

    def UpdateOrders(self):
        for order in self.free_orders:
            order.next()
            if self.gamble_iteration > order.off_time:
                self.FreeOrder(order)
        self.get_new_orders()

    def UpdateCouriers(self):
        for courier in self.free_couriers:
            courier.next()
            if self.gamble_iteration > courier.off_time:
                self.FreeCourier(courier)
        self.get_new_couriers()

    def UpdateActiveRoutes(self):
        for active_route in self.active_routes:
            active_route.next()
            self.FreeActiveRoute(active_route)
        
    def FreeOrder(self, order):
        order_idx = self.free_orders.index(order)
        self.finished_orders.append(order)
        self.free_orders.pop(order_idx)

    def FreeCourier(self, courier):
        courier_idx = self.free_couriers.index(courier)
        self.finished_couriers.append(courier)
        self.free_couriers.pop(courier_idx)

    def FreeActiveRoute(self, active_route):
        self.finished_orders.append(active_route.order)
        if self.gamble_iteration > active_route.courier.off_time:
            self.finished_couriers.append(active_route.courier)
        else:
            self.free_couriers.append(active_route.courier)
        self.active_routes.pop(self.active_routes.index(active_route))

    def get_new_orders(self):
        for _ in range(self.num_orders_every_gamble):
            self.free_orders.append(get_random_order(self.corner_bounds))

    def get_new_couriers(self):
        pass


    # def get_propositions(self):
    #     return self.dispatch(self.busy_couriers_with_time_of_complite, self.free_couriers, self.unassigned_orders)

    # def assign_couriers(self, propositions: List[Tuple[Courier, Route]]):
    #     for courier, route in propositions:
    #         route_complite_time = self.gamble_iteration_to_time() + distance_to_time(route.distance)
    #         heapq.heappush(self.busy_couriers_with_time_of_complite, (route_complite_time, courier))
    
    # def unassign_couriers(self):
    #     while self.busy_couriers_with_time_of_complite:
    #         complite_time, _ = self.busy_couriers_with_time_of_complite[0]
    #         if complite_time > self.gamble_iteration_to_time():
    #             break
    #         _, courier = heapq.heappop(self.busy_couriers_with_time_of_complite)
    #         self.free_couriers.append(courier)        

    def Next(self):
        self.gamble_iteration += 1
        self.Update()
        self.Assign()

    def GetMetrics(self) -> Dict:
        return {}

