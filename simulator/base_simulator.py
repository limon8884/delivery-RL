from typing import List, Tuple, Dict
from utils import *
import numpy as np
import heapq
import random
import json

from objects.point import Point, get_random_point
from objects.order import Order
from objects.courier import Courier
from objects.active_route import ActiveRoute
from objects.gamble_triple import GambleTriple
from objects.utils import *

class BaseSimulator:
    def __init__(self, dispatch, step=0.5) -> None:
        self.env_config = None
        self.dispatch = dispatch
        self.step = step
        self.SetParams()

        self.gamble_iteration = 0
        self.total_gamble_eta = 0

        self.free_orders = []
        self.free_couriers = []
        self.active_routes = []

        self.finished_couriers = []
        self.finished_orders = []

        self.InitCouriers()
        self.current_reward = 0

    def InitCouriers(self):
        for _ in range(self.env_config['num_couriers']):
            courier = Courier(
                get_random_point(self.corner_bounds),
                self.gamble_iteration,
                self.gamble_iteration + self.env_config['courier_live_time_gambles']
            )
            self.free_couriers.append(courier)

    def SetParams(self):
        with open('configs/simulator_settings.json') as f:
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
        assignments = self.dispatch(GambleTriple(self.free_orders, self.free_couriers, self.active_routes))
        self.UpdateAssignment(assignments)

        self.total_gamble_eta = 0
        for o_idx, c_idx in assignments:
            o = self.free_orders[o_idx]
            c = self.free_couriers[c_idx]

            if o is None or c is None:
                continue

            self.active_routes.append(ActiveRoute(c, o, self.gamble_iteration))
            self.free_orders[o_idx] = None
            self.free_couriers[c_idx] = None
            self.total_gamble_eta += distance(c.position, o.point_from)

        self.free_orders = [o for o in self.free_orders if o is not None]
        self.free_couriers = [c for c in self.free_couriers if c is not None]
        

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
        self.UpdateReward()

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

    def UpdateReward(self, reward=1):
        self.current_reward += reward

    def UpdateAssignment(self, assignment):
        self.last_assignment = assignment

    def Next(self):
        assert self.env_config is not None
        self.gamble_iteration += 1
        self.Assign()
        self.Update()
    
    def GetReward(self):
        r = self.current_reward
        self.current_reward = 0
        return r
    
    def GetAssignment(self):
        return self.last_assignment

    def GetState(self):
        return GambleTriple(self.free_orders, self.free_couriers, self.active_routes)

    def GetMetrics(self) -> Dict:
        return {
            'iter': self.gamble_iteration,
            'completed_orders': sum([int(order.is_completed) for order in self.finished_orders]),
            'finished_orders': len(self.finished_orders),
            'current_free_couriers': len(self.free_couriers),
            'current_free_orders': len(self.free_orders),
            'current_active_routes': len(self.active_routes),
            'total_eta': self.total_gamble_eta
        }


class ManualSimulator(BaseSimulator):
    def __init__(self, step=0.5) -> None:
        self.env_config = None
        self.step = step
        self.SetParams()

        self.gamble_iteration = 0
        self.total_gamble_eta = 0

        self.free_orders = []
        self.free_couriers = []
        self.active_routes = []

        self.finished_couriers = []
        self.finished_orders = []

        self.InitCouriers()
        self.current_reward = 0

    def Assign(self, assignments):
        self.UpdateAssignment(assignments)

        self.total_gamble_eta = 0
        for o_idx, c_idx in assignments:
            o = self.free_orders[o_idx]
            c = self.free_couriers[c_idx]

            if o is None or c is None:
                continue

            self.active_routes.append(ActiveRoute(c, o, self.gamble_iteration))
            self.free_orders[o_idx] = None
            self.free_couriers[c_idx] = None
            self.total_gamble_eta += distance(c.position, o.point_from)

        self.free_orders = [o for o in self.free_orders if o is not None]
        self.free_couriers = [c for c in self.free_couriers if c is not None]

    def Next(self, assignments):
        assert self.env_config is not None
        self.gamble_iteration += 1
        self.Assign(assignments)
        self.Update()