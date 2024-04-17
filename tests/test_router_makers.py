from datetime import datetime, timedelta

from src.objects import Claim, Point, Courier, Route, Order
from src.router_makers import AppendRouteMaker
# from src.utils import get_random_point


BASE_DTTM = datetime.utcnow()


def make_claim(source_point: Point, destination_point: Point) -> Claim:
    return Claim(
        0,
        source_point=source_point,
        destination_point=destination_point,
        creation_dttm=BASE_DTTM,
        cancell_if_not_assigned_dttm=BASE_DTTM + timedelta(days=1),
        waiting_on_point_source=timedelta(seconds=0),
        waiting_on_point_destination=timedelta(seconds=0)
    )


def test_append_router():
    router = AppendRouteMaker(max_points_lenght=8, cutoff_radius=0.0)
    crr = Courier(0, Point(1, 0), BASE_DTTM, BASE_DTTM + timedelta(days=1), 'auto')

    cl1 = make_claim(Point(0, 0), Point(0, 1))
    rt = Route.from_claim(cl1)

    cl2 = make_claim(Point(0, 0.5), Point(0, 1.5))
    router.add_claim(rt, crr.position, cl2)

    assert len(rt.route_points) == 4
    for rp, e in zip(rt.route_points, [0, 0.5, 1.0, 1.5]):
        assert rp.point.y == e
    assert rt.distance() == 1.5
    assert rt.distance_with_arrival(crr.position) == 2.5

    cl3 = make_claim(Point(0, 2), Point(0, 3))
    router.add_claim(rt, crr.position, cl3)
    assert rt.distance() == 3.0

    cl4 = make_claim(Point(1, 0), Point(0.5, 0))
    router.add_claim(rt, crr.position, cl4)
    assert rt.distance() == 4.0
    assert rt.distance_with_arrival(crr.position) == 4.0
    assert len(rt.route_points) == 8
    for rp, x, y in zip(rt.route_points, [1, 0.5, 0, 0, 0, 0], [0, 0, 0, 0.5, 1.0, 1.5]):
        assert rp.point.x == x
        assert rp.point.y == y


def test_append_router_cutoff_radius_activates():
    router = AppendRouteMaker(max_points_lenght=8, cutoff_radius=0.5)
    crr = Courier(0, Point(0, 0), BASE_DTTM, BASE_DTTM + timedelta(days=1), 'auto')

    cl1 = make_claim(Point(0, 1.0), Point(0, 100.0))
    rt = Route.from_claim(cl1)
    ord = Order(0, BASE_DTTM, crr, rt, [])
    ord.next(BASE_DTTM + timedelta(seconds=1), speed=0.6)

    cl2 = make_claim(Point(0, 0.7), Point(0, 101.0))
    router.add_claim(rt, crr.position, cl2)

    assert len(rt.route_points) == 4
    for rp, e in zip(rt.route_points, [1.0, 0.7, 100.0, 101.0]):
        assert rp.point.y == e


def test_append_router_cutoff_radius_not_activates():
    router = AppendRouteMaker(max_points_lenght=8, cutoff_radius=0.5)
    crr = Courier(0, Point(0, 0), BASE_DTTM, BASE_DTTM + timedelta(days=1), 'auto')

    cl1 = make_claim(Point(0, 1.0), Point(0, 100.0))
    rt = Route.from_claim(cl1)
    ord = Order(0, BASE_DTTM, crr, rt, [])
    ord.next(BASE_DTTM + timedelta(seconds=1), speed=0.4)

    cl2 = make_claim(Point(0, 0.7), Point(0, 101.0))
    router.add_claim(rt, crr.position, cl2)

    assert len(rt.route_points) == 4
    for rp, e in zip(rt.route_points, [0.7, 1.0, 100.0, 101.0]):
        assert rp.point.y == e
