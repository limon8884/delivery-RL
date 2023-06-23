from typing import List, Tuple, Dict, Sequence
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

from simulator.item_generators import CourierGenerator, OrderGenerator, ActiveRouteGenerator
from simulator.timer import Timer


class Index:
    '''
    A structure for tracking couriers or orders
    '''
    def __init__(self) -> None:
        self.data = {}

    def insert(self, item):
        assert hasattr(item, 'id')
        assert self.data.get(item.id) is None, 'ID is in Index now'
        self.data[item.id] = item        

    def erase(self, id):
        assert isinstance(id, int), 'Input should be ID (int)'
        assert self.data.get(id) is not None, 'ID is not in Index'
        del self.data[id]

    def items(self):
        return list(self.data.values())
    
    def get(self, id):
        assert isinstance(id, int), 'Input should be ID (int)'
        assert self.data.get(id) is not None, 'ID is not in Index'
        return self.data[id]
    

class Simulator:
    def __init__(self, step=0.5, seed=None) -> None:
        self.env_config = None
        self.step = step
        if seed is not None:
            self.SetSeed(seed)
        self.Initialize()

        self.finished_couriers = []
        self.finished_orders = []

        self.InitOrders()
        self.InitCouriers()   

    def SetSeed(self, seed): 
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def Initialize(self):
        with open('configs/simulator_settings.json') as f:
            self.env_config = json.load(f)
            
        self.timer = Timer()

        self.corner_bounds = (
            Point(self.env_config['bounds']['left'], self.env_config['bounds']['lower']),
            Point(self.env_config['bounds']['right'], self.env_config['bounds']['upper'])
        )
        self.order_generator = OrderGenerator(corner_bounds=self.corner_bounds, timer=self.timer, order_live_time=self.env_config['order_live_time_gambles'])
        self.courier_generator = CourierGenerator(corner_bounds=self.corner_bounds, timer=self.timer, courier_live_time=self.env_config['courier_live_time_gambles'])
        self.active_route_generator = ActiveRouteGenerator(corner_bounds=self.corner_bounds, timer=self.timer)

        self.free_orders = Index()
        self.free_couriers = Index()
        self.active_routes = Index()

        self.gamble_info = {
            'iteration': 0,
            'total_eta': 0,
            'reward': 0
        }

    def Next(self, assignments: Sequence[Tuple[int, int]]):
        '''
        Attention! assignments - is a sequence of ID's, not indexes
        '''
        self.gamble_info['iteration'] += 1
        self.Assign(assignments)
        self.Update()
        self.timer.Update()

    def Update(self):
        self.UpdateOrders()
        self.UpdateCouriers()
        self.UpdateActiveRoutes()

    def Assign(self, assignments):
        self.last_assignment = assignments
        self.gamble_info['total_eta'] = 0

        for o_id, c_id in assignments:
            o = self.free_orders.get(o_id)
            c = self.free_couriers.get(c_id)
            ar = self.active_route_generator(o, c)
            self.active_routes.insert(ar)
            self.free_orders.erase(o_id)
            self.free_couriers.erase(c_id)
            self.gamble_info['total_eta'] += distance(c.position, o.point_from)

    def InitCouriers(self):
        for _ in range(self.env_config['start_num_couriers']):
            courier = self.courier_generator()
            self.free_couriers.insert(courier)
    
    def InitOrders(self):
        for _ in range(self.env_config['start_num_orders']):
            order = self.order_generator()
            self.free_orders.insert(order)
        
    def UpdateOrders(self):
        for order in self.free_orders.items():
            order.next(self.timer())
            if self.timer() > order.expire_time:
                self.FreeOrder(order)
        self.GetNewOrders()

    def UpdateCouriers(self):
        for courier in self.free_couriers.items():
            courier.next(self.timer())
            if self.timer() > courier.off_time:
                self.FreeCourier(courier)
        self.GetNewCouriers()

    def UpdateActiveRoutes(self):
        for active_route in self.active_routes.items():
            active_route.next(self.step, self.timer())
            if not active_route.is_active:
                self.FreeActiveRoute(active_route)
        
    def FreeOrder(self, order: Order):
        self.free_orders.erase(order.id)
        self.finished_orders.append(order)

    def FreeCourier(self, courier: Courier):
        self.free_couriers.erase(courier.id)
        self.finished_couriers.append(courier)

    def FreeActiveRoute(self, active_route: ActiveRoute):
        self.finished_orders.append(active_route.order)
        if self.timer() > active_route.courier.off_time:
            self.finished_couriers.append(active_route.courier)
        else:
            self.free_couriers.insert(active_route.courier)
        self.active_routes.erase(active_route.id)

    def GetNewOrders(self):
        for _ in range(self.env_config['num_orders_every_gamble']):
            order = self.order_generator()
            self.free_orders.insert(order)

    def GetNewCouriers(self):
        for _ in range(self.env_config['num_couriers_every_gamble']):
            courier = self.courier_generator()
            self.free_couriers.insert(courier)

    def GetState(self):
        return GambleTriple(self.free_orders.items(), self.free_couriers.items(), self.active_routes.items())

    def GetMetrics(self) -> Dict:
        return {
            'iter': self.gamble_info['iteration'],
            'completed_orders': sum([int(order.is_completed) for order in self.finished_orders]),
            'finished_orders': len(self.finished_orders),
            'current_free_couriers': len(self.free_couriers.items()),
            'current_free_orders': len(self.free_orders.items()),
            'current_active_routes': len(self.active_routes.items()),
            'total_eta': self.gamble_info['total_eta']
        }



