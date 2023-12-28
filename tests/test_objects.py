from pytest import approx
import pytest
from datetime import datetime, timedelta


from src_new.utils import (
    get_random_point
)
from src_new.objects import (
    Point,
    Route,
    Courier,
    Claim,
    Order,
)
from src_new.database.database import Database, TableName, Event

CREATION_TIME = datetime.utcnow()


def test_point():
    p1 = Point(0, 1)
    p2 = Point(1, 2)
    assert p1 + p2 == Point(1, 3)
    assert p2 - p1 == Point(1, 1)
    assert p1 * 10.0 == Point(0, 10)
    assert p2 * 10 == Point(10, 20)

    zero = Point(0, 0)
    for _ in range(10):
        p = get_random_point()
        assert Point.distance(zero, p) == approx((p.x**2 + p.y**2)**0.5, 0.0001)
        assert Point.distance(zero, p.normalize()) == approx(1, 0.001)


def test_route():
    p1 = Point(0, 1)
    p2 = Point(1, 1)
    p3 = Point(1, 1)
    p4 = Point(1, 0)
    p5 = Point(0, 0)
    assert Route([p1, p2, p3, p4, p5], [], []).distance() == 3
    assert Route([p1, p1], [], []).distance() == 0
    assert Route([p5, p4, p3, p1, p5], [], []).distance() == 4
    assert Route([p1, p4], [], []).distance() == approx(2**0.5, 0.001)


def test_courier():
    crr = Courier(0, Point(0, 0), CREATION_TIME, CREATION_TIME + timedelta(seconds=2), 'auto')
    crr.set_position(Point(1, 1))
    assert crr.position == Point(1, 1)
    crr.next(CREATION_TIME + timedelta(seconds=2))
    assert crr.done()
    with pytest.raises(RuntimeError):
        crr.next(CREATION_TIME + timedelta(seconds=3))
    # assert crr.done_dttm() == CREATION_TIME + timedelta(seconds=2)


def test_claim():
    clm = Claim(0, Point(0, 0), Point(1, 1),
                CREATION_TIME, CREATION_TIME + timedelta(seconds=10), timedelta(seconds=1), timedelta(seconds=1))
    clm.next(CREATION_TIME + timedelta(seconds=9))
    assert not clm.done()
    assert clm.status is Claim.Status.UNASSIGNED
    clm.next(CREATION_TIME + timedelta(seconds=11))
    assert clm.done()
    # assert clm.done_dttm() == CREATION_TIME + timedelta(seconds=11)


@pytest.mark.parametrize("n_claims, wait_secs, speed", [(1, 0, 0.5), (5, 0, 1.0), (2, 1, 0.333), (10, 10, 0.5)])
def test_order(n_claims: int, wait_secs: int, speed: float):
    crr = Courier(0, Point(0, 0), CREATION_TIME, CREATION_TIME + timedelta(days=1), 'auto')
    for _ in range(100):
        current_time = CREATION_TIME
        claims = [
            Claim(i, get_random_point(), get_random_point(),
                  CREATION_TIME, CREATION_TIME + timedelta(days=1),
                  timedelta(seconds=wait_secs), timedelta(seconds=wait_secs))
            for i in range(n_claims)
            ]
        rt = Route(
            [c.source_point for c in claims] + [c.destination_point for c in claims],
            [c.id for c in claims] + [c.id for c in claims],
            [Route.PointType.SOURCE] * n_claims + [Route.PointType.DESTINATION] * n_claims
        )
        # rt = Route([get_random_point() for _ in range(n_points)], [], [])
        total_dist = Point.distance(crr.position, rt.points[0]) + rt.distance()
        ord = Order(0, CREATION_TIME, crr, rt, claims)
        assert ord.status is Order.Status.IN_ARRIVAL
        total_time = 2 * n_claims * wait_secs + total_dist / speed

        for t in range(int(total_time)):
            current_time += timedelta(seconds=1)
            ord.next(current_time, speed=speed)
        assert ord.status is not Order.Status.COMPLETED
        current_time += timedelta(seconds=1)
        ord.next(current_time, speed=speed)
        assert ord.status is Order.Status.COMPLETED
        # assert ord.done_dttm() == current_time


def test_with_db(tmp_path):
    db = Database(tmp_path / 'testbase.db')
    crr = Courier(0, Point(0, 0), CREATION_TIME, CREATION_TIME + timedelta(seconds=2), 'auto', db=db)
    crr.next(CREATION_TIME + timedelta(seconds=1))
    crr.next(CREATION_TIME + timedelta(seconds=2))
    assert crr.done()

    res = db.select(f'''select courier_id, event from {TableName.COURIER_TABLE.value}''')
    assert res[0] == (0, Event.COURIER_STARTED.value)
    assert res[1] == (0, Event.COURIER_ENDED.value)
    assert len(res) == 2
