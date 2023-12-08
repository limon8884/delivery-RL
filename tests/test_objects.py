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
    assert Route([p1, p2, p3, p4, p5]).distance() == 3
    assert Route([p1, p1]).distance() == 0
    assert Route([p5, p4, p3, p1, p5]).distance() == 4
    assert Route([p1, p4]).distance() == approx(2**0.5, 0.001)


def test_courier():
    crr = Courier(0, Point(0, 0), datetime.utcnow(), CREATION_TIME, 'auto')
    crr.set_position(Point(1, 1))
    assert crr.position == Point(1, 1)
    crr.next(CREATION_TIME)
    assert crr.done()


def test_claim():
    clm = Claim(0, Point(0, 0), Point(1, 1),
                datetime.utcnow(), CREATION_TIME + timedelta(seconds=10), timedelta(seconds=1))
    clm.next(CREATION_TIME + timedelta(seconds=9))
    assert not clm.done()
    clm.next(CREATION_TIME + timedelta(seconds=11))
    assert clm.done()


@pytest.mark.parametrize("n_points, wait_secs, speed", [(1, 0, 0.5), (5, 0, 1.0), (2, 1, 0.333), (10, 10, 0.5)])
def test_order(n_points: int, wait_secs: int, speed: float):
    crr = Courier(0, Point(0, 0), datetime.utcnow(), CREATION_TIME + timedelta(days=1), 'auto')
    for _ in range(100):
        current_time = CREATION_TIME
        rt = Route([get_random_point() for _ in range(n_points)])
        total_dist = Point.distance(crr.position, rt.points[0]) + rt.distance()
        ord = Order(0, CREATION_TIME, crr, rt, timedelta(seconds=wait_secs), [])
        assert ord.status is Order.Status.IN_ETA
        total_time = n_points * wait_secs + total_dist / speed

        for t in range(int(total_time)):
            current_time += timedelta(seconds=1)
            ord.next(current_time, speed=speed)
        assert ord.status is not Order.Status.COMPLETED
        current_time += timedelta(seconds=1)
        ord.next(current_time, speed=speed)
        assert ord.status is Order.Status.COMPLETED
