import torch
# import numpy as np

from src_new.dispatchs.base_dispatch import BaseDispatch
# from src_new.dispatchs.scorers import BaseScorer
from src_new.objects import (
    Order,
    Gamble,
    Assignment,
)
from src_new.networks.encoders import GambleEncoder
from src_new.networks.bodies import BaseSequentialDispatchNetwork


class NeuralSequantialDispatch(BaseDispatch):
    def __init__(self, encoder: GambleEncoder, network: BaseSequentialDispatchNetwork, **kwargs) -> None:
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.max_num_points_in_route = kwargs['max_num_points_in_route']

    def __call__(self, gamble: Gamble) -> Assignment:
        assignment_list = []
        gamble_embs_dict = self.encoder(gamble)
        courier_order_embs = gamble_embs_dict['couriers'] + gamble_embs_dict['orders']
        n_couriers = len(gamble.couriers)
        n_orders = len(gamble.orders)
        num_points_list = [0] * n_couriers + [len(order.route.route_points) for order in gamble.orders]
        for claim_idx, claim_emb in enumerate(gamble_embs_dict['claims']):
            probas = self.network(claim_emb, courier_order_embs)
            mask = _make_is_full_mask(num_points_list, self.max_num_points_in_route)
            choice = int(torch.argmax(probas + mask).item())
            if choice == n_couriers + n_orders:
                continue
            num_points_list[choice] += 2
            if choice < n_couriers:
                assignment_list.append((
                    gamble.couriers[choice].id,
                    gamble.claims[claim_idx].id
                ))
            else:
                assignment_list.append((
                    gamble.orders[choice - n_couriers].courier.id,
                    gamble.claims[claim_idx].id
                ))
        return Assignment(assignment_list)


def _make_is_full_mask(num_points_list: list[int], max_num_points_in_route: int) -> torch.FloatTensor:
    is_full_mask = [n_points >= max_num_points_in_route - 1 for n_points in num_points_list] + [False]
    return torch.FloatTensor(is_full_mask) * -torch.inf
