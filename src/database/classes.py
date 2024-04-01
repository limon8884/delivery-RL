from enum import Enum


class TableName(Enum):
    COURIER_TABLE = 'couriers'
    CLAIM_TABLE = 'claims'
    ORDER_TABLE = 'orders'


class Metric(Enum):
    CR = 'cr'
    CTD = 'ctd'
    NOT_BATCHED_ARRIVAL_DISTANCE = 'not_batched_arrival_distance'
    NUM_COURIERS = 'avg_num_couriers_in_gamble'
    NUM_CLAIMS = 'avg_num_claims_in_gamble'
    NUM_ORDERS = 'avg_num_orders_in_gamble'
    DEBUG = 'debug'


class Event(Enum):
    COURIER_STARTED = 'courier_started'
    COURIER_ENDED = 'courier_ended'
    CLAIM_CREATED = 'claim_created'
    CLAIM_ASSIGNED = 'claim_assigned'
    CLAIM_CANCELLED = 'claim_cancelled'
    CLAIM_COMPLETED = 'claim_completed'
    ORDER_CREATED = 'order_created'
    ORDER_FINISHED = 'order_completed'
