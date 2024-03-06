from src.objects import Route, Point, Claim


class BaseRouteMaker:
    def __init__(self, max_points_lenght: int, cutoff_radius: float) -> None:
        self.max_points_lenght = max_points_lenght
        self.cutoff_radius = cutoff_radius

    def add_claim(self, route: Route, courier_position: Point, new_claim: Claim) -> None:
        '''
        Updates route with given new claim
        '''
        raise NotImplementedError


class AppendRouteMaker(BaseRouteMaker):
    def add_claim(self, route: Route, courier_position: Point, new_claim: Claim) -> None:
        if len(route.route_points) >= self.max_points_lenght:
            raise RuntimeError("Max points limit reached")
        length = len(route.route_points)
        source_route_point = Route.RoutePoint(new_claim.source_point, new_claim.id, Route.PointType.SOURCE)
        destination_route_point = Route.RoutePoint(new_claim.destination_point,
                                                   new_claim.id, Route.PointType.DESTINATION)
        next_point_idx = 1 \
            if Point.distance(route.next_route_point().point, courier_position) < self.cutoff_radius \
            else 0

        best_idxs: tuple[int, int] = (-1, -1)
        best_dist = 1e10
        for new_source_idx in range(next_point_idx, length + 1):
            route.route_points.insert(new_source_idx, source_route_point)
            for new_destination_idx in range(new_source_idx + 1, length + 2):
                route.route_points.insert(new_destination_idx, destination_route_point)
                dist = route.distance_with_arrival(courier_position)
                if dist < best_dist:
                    best_dist = dist
                    best_idxs = (new_source_idx, new_destination_idx)
                del route.route_points[new_destination_idx]
            del route.route_points[new_source_idx]

        route.route_points.insert(best_idxs[0], source_route_point)
        route.route_points.insert(best_idxs[1], destination_route_point)
