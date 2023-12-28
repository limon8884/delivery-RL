import sqlite3
from pathlib import Path
from datetime import datetime
# from enum import Enum
from src_new.database.classes import TableName, Event, Metric
from src_new.database.sql_metrics_queries import (
    _sql_query_cr,
    _sql_query_ctd
)


class Database:
    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        # self._connection.set_trace_callback(print)
        self._cursor = self._connection.cursor()
        self._cursor.execute(f'CREATE TABLE IF NOT EXISTS {TableName.COURIER_TABLE.value} (courier_id, dttm, event)')
        self._cursor.execute(f'CREATE TABLE IF NOT EXISTS {TableName.CLAIM_TABLE.value} (claim_id, dttm, event)')
        self._cursor.execute(f'CREATE TABLE IF NOT EXISTS {TableName.ORDER_TABLE.value} (order_id, dttm, event)')

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def insert(self, table_name: TableName, item_id: int, dttm: datetime, event: Event):
        sql = f'INSERT INTO {table_name.value} VALUES (?, ?, ?)'
        self._cursor.execute(sql, (item_id, dttm, event.value))

    def select(self, sql: str, select_one=False):
        cursor = self._connection.cursor()
        res = cursor.execute(sql)
        if select_one:
            return res.fetchone()
        res_list = res.fetchall()
        cursor.close()
        return res_list

    def get_metric(self, metric: Metric) -> float:
        if metric is Metric.CR:
            return self.select(_sql_query_cr(), select_one=True)[0]
        elif metric is Metric.CTD:
            return self.select(_sql_query_ctd(), select_one=True)[0]
        else:
            raise RuntimeError('Metric not implemented')
