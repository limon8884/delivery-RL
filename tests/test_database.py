# import pytest

from src_new.database.database import Database, TableName, Metric, Event
from datetime import datetime, timedelta


BASE_DTTM = datetime.utcnow()


def test_creation(tmp_path):
    db = Database(tmp_path / 'testbase.db')
    db.commit()

    db.insert(TableName.COURIER_TABLE, 0, BASE_DTTM, Event.CLAIM_CREATED)
    db.insert(TableName.CLAIM_TABLE, 1, BASE_DTTM, Event.CLAIM_CANCELLED)

    res = db.select(f'''
    select *
    from {TableName.CLAIM_TABLE.value}
    ''')

    assert len(res) == 1 and len(res[0]) == 3
    assert res[0][0] == 1
    assert res[0][2] == Event.CLAIM_CANCELLED.value
    db.commit()


def test_metrics(tmp_path):
    db = Database(tmp_path / 'testbase.db')
    db.commit()

    db.insert(TableName.CLAIM_TABLE, 0, BASE_DTTM, Event.CLAIM_CREATED)
    db.insert(TableName.CLAIM_TABLE, 0, BASE_DTTM + timedelta(seconds=1), Event.CLAIM_COMPLETED)
    db.insert(TableName.CLAIM_TABLE, 1, BASE_DTTM, Event.CLAIM_CREATED)
    db.insert(TableName.CLAIM_TABLE, 1, BASE_DTTM + timedelta(seconds=2), Event.CLAIM_COMPLETED)
    db.insert(TableName.CLAIM_TABLE, 2, BASE_DTTM, Event.CLAIM_CREATED)
    db.insert(TableName.CLAIM_TABLE, 2, BASE_DTTM + timedelta(seconds=3), Event.CLAIM_CANCELLED)
    db.insert(TableName.CLAIM_TABLE, 3, BASE_DTTM, Event.CLAIM_CREATED)
    db.insert(TableName.CLAIM_TABLE, 3, BASE_DTTM + timedelta(seconds=3), Event.CLAIM_COMPLETED)

    assert db.get_metric(Metric.CR) == 0.75
    assert db.get_metric(Metric.CTD) == 2.0
