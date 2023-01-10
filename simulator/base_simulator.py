from typing import List, Tuple, Dict
from utils import *
import numpy as np
import heapq
import random
import json

class BaseSimulator:
    def __init__(self, dispatch, step=0.5) -> None:
        self.env_config = None
        self.dispatch = dispatch
        self.step = step
        self.SetParams()

        self.gamble_iteration = 0

        self.free_orders = []
        self.free_couriers = []
        self.active_routes = []

        self.finished_couriers = []
        self.finished_orders = []

        self.InitCouriers()

    def InitCouriers(self):
        for _ in range(self.env_config['num_couriers']):
            courier = Courier(
                get_random_point(self.corner_bounds),
                self.gamble_iteration,
                self.gamble_iteration + self.env_config['courier_live_time_gambles']
            )
            self.free_couriers.append(courier)

    def SetParams(self):
        with open('environment_config.json') as f:
            self.env_config = json.load(f)
            
        self.corner_bounds = (
            Point(self.env_config['bounds']['left'], self.env_config['bounds']['lower']),
            Point(self.env_config['bounds']['right'], self.env_config['bounds']['upper'])
        )

    def Update(self):
        self.UpdateOrders()
        self.UpdateCouriers()
        self.UpdateActiveRoutes()

    def Assign(self):
        assignments = self.dispatch(self.free_orders, self.free_couriers, self.active_routes)

        for o_idx, c_idx in assignments:
            o = self.free_orders[o_idx]
            c = self.free_couriers[c_idx]
            self.active_routes.append(ActiveRoute(c, o, self.gamble_iteration))

        for o_idx, _ in sorted(assignments, key=lambda x: x[0], reverse=True):
            o = self.free_orders.pop(o_idx)
        
        for _, c_idx in sorted(assignments, key=lambda x: x[1], reverse=True):
            c = self.free_couriers.pop(c_idx)
        

    def UpdateOrders(self):
        for order in self.free_orders:
            order.next(self.gamble_iteration)
            if self.gamble_iteration > order.expire_time:
                self.FreeOrder(order)
        self.GetNewOrders()

    def UpdateCouriers(self):
        for courier in self.free_couriers:
            courier.next(self.gamble_iteration)
            if self.gamble_iteration > courier.off_time:
                self.FreeCourier(courier)
        self.GetNewCouriers()

    def UpdateActiveRoutes(self):
        for active_route in self.active_routes:
            active_route.next(self.step, self.gamble_iteration)
            if not active_route.is_active:
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

    def GetNewOrders(self):
        for _ in range(self.env_config['num_orders_every_gamble']):
            order = Order(
                get_random_point(self.corner_bounds), 
                get_random_point(self.corner_bounds), 
                self.gamble_iteration,
                self.gamble_iteration + self.env_config['order_live_time_gambles']
            )
            self.free_orders.append(order)

    def GetNewCouriers(self):
        pass

    def Next(self):
        assert self.env_config is not None
        self.gamble_iteration += 1
        self.Update()
        self.Assign()

    def GetMetrics(self) -> Dict:
        return {
            'iter': self.gamble_iteration,
            'completed_orders': sum([int(order.is_completed) for order in self.finished_orders]),
            'finished_orders': len(self.finished_orders),
            'current_free_couriers': len(self.free_couriers),
            'current_free_orders': len(self.free_orders),
            'current_active_routes': len(self.active_routes)
        }
