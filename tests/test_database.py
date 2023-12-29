# import pytest

from src_new.database.database import Database, TableName, Metric, Event
from src_new.database.logger import Logger
from datetime import datetime, timedelta


BASE_DTTM = datetime.utcnow()


def test_creation(tmp_path):
    db = Database(tmp_path / 'testbase.db')
    logger = Logger(run_id=-1)

    logger.insert(TableName.COURIER_TABLE, 0, BASE_DTTM, Event.CLAIM_CREATED)
    logger.insert(TableName.CLAIM_TABLE, 1, BASE_DTTM, Event.CLAIM_CANCELLED)
    db.export_from_logger(logger)

    res = db.select(f'''
    select *
    from {TableName.CLAIM_TABLE.value}
    ''')

    assert len(res) == 1 and len(res[0]) == 4
    assert res[0][1] == 1
    assert res[0][3] == Event.CLAIM_CANCELLED.value


def test_metrics(tmp_path):
    db = Database(tmp_path / 'testbase.db')
    logger = Logger(run_id=-1)

    logger.insert(TableName.CLAIM_TABLE, 0, BASE_DTTM, Event.CLAIM_CREATED)
    logger.insert(TableName.CLAIM_TABLE, 0, BASE_DTTM + timedelta(seconds=1), Event.CLAIM_COMPLETED)
    logger.insert(TableName.CLAIM_TABLE, 1, BASE_DTTM, Event.CLAIM_CREATED)
    logger.insert(TableName.CLAIM_TABLE, 1, BASE_DTTM + timedelta(seconds=2), Event.CLAIM_COMPLETED)
    logger.insert(TableName.CLAIM_TABLE, 2, BASE_DTTM, Event.CLAIM_CREATED)
    logger.insert(TableName.CLAIM_TABLE, 2, BASE_DTTM + timedelta(seconds=3), Event.CLAIM_CANCELLED)
    logger.insert(TableName.CLAIM_TABLE, 3, BASE_DTTM, Event.CLAIM_CREATED)
    logger.insert(TableName.CLAIM_TABLE, 3, BASE_DTTM + timedelta(seconds=3), Event.CLAIM_COMPLETED)

    db.export_from_logger(logger)
    assert db.get_metric(Metric.CR, run_id=-1) == 0.75
    assert db.get_metric(Metric.CTD, run_id=-1) == 2.0
