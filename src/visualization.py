import numpy as np
import matplotlib.pyplot as plt
import imageio
import typing
from pathlib import Path
from matplotlib.lines import Line2D
from colour import Color


from src.objects import (
    Gamble,
    Assignment,
    Claim,
    Courier,
    Order,
    Point
)


class Visualization:
    def __init__(self, save_path: Path, mute: bool = False, duration_sec=1, figsize=(10, 10)) -> None:
        self.images = []
        self.mute = mute
        self.save_path = save_path
        self.duration_sec = duration_sec
        self.figsize = figsize
        self.config = {}

    def visualize(self, gamble: Gamble, assignment: Assignment, step: int):
        if self.mute:
            return
        fig, ax = plt.subplots()
        ax.set_title(f'Step {step}')
        plot_claims(gamble.claims, ax, self.config['claim'])
        plot_couriers(gamble.couriers, ax, self.config['courier'])
        plot_orders(gamble.orders, ax, self.config['order'])
        plot_assignment(gamble, assignment, ax, self.config['assignment'])
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        self.images.append(image.reshape(self.figsize[1] * 100, self.figsize[0] * 100, 3))

    def save_visualization(self):
        imageio.mimsave(self.save_path, self.images, format='GIF', duration=int(self.duration_sec * 1000))
        self.images = []


def plot_assignment(gamble: Gamble, assignment: Assignment, ax: typing.Any, cfg: dict) -> None:
    pass


def make_get_color(cfg: dict):
    from_color = Color(cfg['min']['color'])
    to_color = Color(cfg['max']['color'])
    palette = [c.hex for c in from_color.range_to(to_color, cfg['num_pieces'])]
    min_value = cfg['min']['value']
    max_value = cfg['max']['value']

    def func(value):
        value = np.clip(value, min_value, max_value)
        piece_idx = int((value - min_value) / (max_value - min_value) * (cfg['num_pieces'] - 1))
        return palette[piece_idx]

    return func


def plot_claims(claims: list[Claim], ax: typing.Any, cfg: dict):
    get_color = make_get_color(cfg['gradient'])
    x_source_coords, y_source_coords = [], []
    x_destination_coords, y_destination_coords = [], []
    colors = []
    annd = cfg['annotation_deltas']
    for claim in claims:
        x_source_coords.append(claim.source_point.x)
        y_source_coords.append(claim.source_point.y)
        x_destination_coords.append(claim.destination_point.x)
        y_destination_coords.append(claim.destination_point.y)
        colors.append(get_color(claim.time))
        ax.annotate(claim.id, (claim.source_point.x + annd['x'], claim.source_point.y + annd['y']))
        ax.annotate(claim.id, (claim.destination_point.x + annd['x'], claim.destination_point.y + annd['y']))
    ax.scatter(x_source_coords, y_source_coords, facecolor=colors, marker=cfg['markers']['source'], **cfg['kwargs'])
    ax.scatter(x_destination_coords, y_destination_coords, facecolor=colors, marker=cfg['markers']['destination'],
               **cfg['kwargs'])


def plot_couriers(couriers: list[Courier], ax: typing.Any, cfg: dict):
    x_pos, y_pos = [], []
    annd = cfg['annotation_deltas']
    for courier in couriers:
        x_pos.append(courier.position.x)
        y_pos.append(courier.position.y)
        ax.annotate(courier.id, (courier.position.x + annd['x'], courier.position.y + annd['y']))
    ax.scatter(x_pos, y_pos, facecolor=cfg['color'], marker=cfg['marker'], **cfg['kwargs'])


def plot_orders(orders: list[Order], ax: typing.Any, cfg: dict):
    x_points, y_points = [], []
    x_crrs, y_crrs = [], []
    for order in orders:
        prev_x, prev_y = order.courier.position.x, order.courier.position.y
        for pt in order.route:
            line = Line2D([prev_x, pt.x], [prev_y, pt.y], **cfg['links'])
            ax.add_line(line)
            x_points.append(pt.x)
            y_points.append(pt.y)
            prev_x, prev_y = pt.x, pt.y
        x_crrs.append(order.courier.position.x)
        y_crrs.append(order.courier.position.y)
    ax.scatter(x_points, y_points, facecolor=cfg['color'], marker=cfg['markers']['source'], **cfg['kwargs'])
    ax.scatter(x_crrs, y_crrs, facecolor=cfg['color'], marker=cfg['markers']['courier'], **cfg['kwargs'])


# if __name__ == '__main__':
#     v = Visualization('test.gif')
#     data1 = np.random.random((8, 10))
#     data2 = np.random.random((8, 10)) * 5
#     for step, (d1, d2) in enumerate(zip(data1, data2)):
#         v.visualize(d1, d2, step)
#     v.save_visualization()


