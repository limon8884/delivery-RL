from enum import Enum


class TableName(Enum):
    COURIER_TABLE = 'couriers'
    CLAIM_TABLE = 'claims'
    ORDER_TABLE = 'orders'


class Metric(Enum):
    CR = 'cr'
    CTD = 'ctd'
    DEBUG = 'debug'


class Event(Enum):
    COURIER_STARTED = 'courier_started'
    COURIER_ENDED = 'courier_ended'
    CLAIM_CREATED = 'claim_created'
    CLAIM_CANCELLED = 'claim_cancelled'
    CLAIM_COMPLETED = 'claim_completed'
    ORDER_CREATED = 'order_created'
    ORDER_FINISHED = 'order_completed'
