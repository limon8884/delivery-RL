import numpy as np
import matplotlib.pyplot as plt
import imageio
import typing
import json
from pathlib import Path
from matplotlib.lines import Line2D
from colour import Color
from PIL import Image


from src.objects import (
    Gamble,
    Assignment,
    Claim,
    Courier,
    Order,
    Route
)


class Visualization:
    def __init__(self, config_path: Path, plot_ords: bool = False, vis_freq: int = 1, figsize=(10, 10)) -> None:
        self.images: list[np.ndarray] = []
        self.plot_ords = plot_ords
        self.figsize = figsize
        self.vis_freq = vis_freq
        with open(config_path) as f:
            self.config = json.load(f)

    def visualize(self, gamble: Gamble, assignment: Assignment, step: int):
        if step % self.vis_freq != 0:
            return
        fig, ax = plt.subplots()
        ax.set_title(f'Step {step}, time {gamble.dttm_start.hour}:{gamble.dttm_start.minute}')
        plot_claims(gamble.claims, ax, self.config['claim'])
        plot_couriers(gamble.couriers, ax, self.config['courier'])
        if self.plot_ords:
            plot_orders(gamble.orders, ax, self.config['order'])
        plot_assignment(gamble, assignment, ax, self.config['assignment'])
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        ncols, nrows = fig.canvas.get_width_height()
        self.images.append(image.reshape(nrows, ncols, 3))
        plt.close(fig)

    def to_gif(self, save_path: Path, duration_sec: int):
        assert len(self.images) > 0
        imageio.mimsave(save_path, self.images, format='GIF', duration=int(duration_sec * 1000))

    def reset(self):
        self.images = []

    def __len__(self):
        return len(self.images)

    def show(self, idx: int):
        assert idx < len(self.images)
        Image.fromarray(self.images[idx]).show()


def plot_assignment(gamble: Gamble, assignment: Assignment, ax: typing.Any, cfg: dict) -> None:
    crrs_coords = {crr.id: crr.position for crr in gamble.couriers}
    clms_coords = {clm.id: clm.source_point for clm in gamble.claims}
    for crr_id, clm_id in assignment.ids:
        if crr_id not in crrs_coords:
            continue
        crr_x, crr_y = crrs_coords[crr_id].x, crrs_coords[crr_id].y
        clm_x, clm_y = clms_coords[clm_id].x, clms_coords[clm_id].y
        line = Line2D([crr_x, clm_x], [crr_y, clm_y], **cfg)
        ax.add_line(line)


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
        colors.append(get_color((claim._dttm - claim.creation_dttm).total_seconds()))
        ax.annotate(claim.id, (claim.source_point.x + annd['x'], claim.source_point.y + annd['y']))
        ax.annotate(claim.id, (claim.destination_point.x + annd['x'], claim.destination_point.y + annd['y']))
    ax.scatter(x_source_coords, y_source_coords, facecolor=colors, **cfg['source'])
    ax.scatter(x_destination_coords, y_destination_coords, facecolor=colors, **cfg['destination'])


def plot_couriers(couriers: list[Courier], ax: typing.Any, cfg: dict):
    x_pos, y_pos = [], []
    annd = cfg['annotation_deltas']
    for courier in couriers:
        x_pos.append(courier.position.x)
        y_pos.append(courier.position.y)
        ax.annotate(courier.id, (courier.position.x + annd['x'], courier.position.y + annd['y']))
    ax.scatter(x_pos, y_pos, facecolor=cfg['color'], marker=cfg['marker'], **cfg['kwargs'])


def plot_orders(orders: list[Order], ax: typing.Any, cfg: dict):
    from_color = Color(cfg['color']['min'])
    to_color = Color(cfg['color']['max'])
    x_source_points, y_source_points = [], []
    x_destination_points, y_destination_points = [], []
    x_crrs, y_crrs = [], []
    source_colors, destination_colors = [], []
    for order in orders:
        palette = [c.hex for c in from_color.range_to(to_color, len(order.claims))]
        claim_id_2_color = {id: c for id, c in zip(order.claims.keys(), palette)}
        prev_x, prev_y = order.courier.position.x, order.courier.position.y
        for rpt in order.route.route_points:
            pt = rpt.point
            line = Line2D([prev_x, pt.x], [prev_y, pt.y], **cfg['links'])
            ax.add_line(line)
            if rpt.point_type is Route.PointType.SOURCE:
                x_source_points.append(pt.x)
                y_source_points.append(pt.y)
                source_colors.append(claim_id_2_color[rpt.claim_id])
            elif rpt.point_type is Route.PointType.DESTINATION:
                x_destination_points.append(pt.x)
                y_destination_points.append(pt.y)
                destination_colors.append(claim_id_2_color[rpt.claim_id])
            prev_x, prev_y = pt.x, pt.y
        x_crrs.append(order.courier.position.x)
        y_crrs.append(order.courier.position.y)
    ax.scatter(x_source_points, y_source_points, facecolor=source_colors, marker=cfg['markers']['source'],
               **cfg['kwargs'])
    ax.scatter(x_destination_points, y_destination_points, facecolor=destination_colors,
               marker=cfg['markers']['destination'], **cfg['kwargs'])
    ax.scatter(x_crrs, y_crrs, facecolor=cfg['color']['courier'], marker=cfg['markers']['courier'], **cfg['kwargs'])
