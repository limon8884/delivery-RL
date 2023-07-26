from src.objects.point import Point


def distance(first_point: Point, second_point: Point):
    return ((first_point.x - second_point.x)**2 + (first_point.y - second_point.y)**2)**0.5
