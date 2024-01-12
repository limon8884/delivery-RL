import sqlite3
from pathlib import Path
from datetime import datetime
# from enum import Enum
from src_new.database.classes import TableName, Event, Metric
from src_new.database.logger import Logger
from src_new.database.sql_metrics_queries import (
    _sql_query_cr,
    _sql_query_ctd,
    _sql_query_num_couriers,
    _sql_query_num_claims,
    _sql_query_num_orders,
)


BUCKET_QUERY_SQLITE_SIZE = 100


class Database:
    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        # self._connection.set_trace_callback(print)
        cursor = self._connection.cursor()
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {TableName.COURIER_TABLE.value} (run_id, courier_id, dttm, event)')
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {TableName.CLAIM_TABLE.value} (run_id, claim_id, dttm, event)')
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {TableName.ORDER_TABLE.value} (run_id, order_id, dttm, event)')
        cursor.close()

    def clear(self):
        cursor = self._connection.cursor()
        cursor.execute(f'DROP TABLE IF EXISTS {TableName.COURIER_TABLE.value}')
        cursor.execute(f'DROP TABLE IF EXISTS {TableName.CLAIM_TABLE.value}')
        cursor.execute(f'DROP TABLE IF EXISTS {TableName.ORDER_TABLE.value}')
        cursor.close()
        self._connection.commit()

    def export_from_logger(self, logger: Logger) -> None:
        self._export_from_logger_table(logger, TableName.COURIER_TABLE)
        self._export_from_logger_table(logger, TableName.CLAIM_TABLE)
        self._export_from_logger_table(logger, TableName.ORDER_TABLE)
        self._connection.commit()

    def _export_from_logger_table(self, logger: Logger, tablename: TableName) -> None:
        if len(logger.data[tablename.value]) == 0:
            return
        cursor = self._connection.cursor()
        for bucket_idx in range(0, len(logger.data[tablename.value]), BUCKET_QUERY_SQLITE_SIZE):
            data = logger.data[tablename.value][bucket_idx:bucket_idx + BUCKET_QUERY_SQLITE_SIZE]
            sql = f'INSERT INTO {tablename.value} VALUES ' + '(?, ?, ?, ?), ' * len(data)
            sql = sql[:-2] + ';\n'
            args = []
            for row in data:
                args.append(logger.run_id)
                args.extend(row)
            cursor.execute(sql, args)

        cursor.close()

    def select(self, sql: str, select_one=False):
        cursor = self._connection.cursor()
        res = cursor.execute(sql)
        if select_one:
            return res.fetchone()
        res_list = res.fetchall()
        cursor.close()
        return res_list

    def get_metric(self, metric: Metric, run_id: int) -> float:
        if metric is Metric.CR:
            return self.select(_sql_query_cr(run_id), select_one=True)[0]
        elif metric is Metric.CTD:
            return self.select(_sql_query_ctd(run_id), select_one=True)[0]
        # elif metric is Metric.NUM_COURIERS:
        #     return self.select(_sql_query_num_couriers(run_id), select_one=True)[0]
        # elif metric is Metric.NUM_CLAIMS:
        #     return self.select(_sql_query_num_claims(run_id), select_one=True)[0]
        # elif metric is Metric.NUM_ORDERS:
        #     return self.select(_sql_query_num_orders(run_id), select_one=True)[0]
        else:
            raise RuntimeError('Metric not implemented')
