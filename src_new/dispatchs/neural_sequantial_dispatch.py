import torch
# import numpy as np

from src_new.dispatchs.base_dispatch import BaseDispatch
# from src_new.dispatchs.scorers import BaseScorer
from src_new.objects import (
    Gamble,
    Assignment,
)
from src_new.networks.encoders import GambleEncoder
from src_new.networks.bodies import BaseSequentialDispatchNetwork


class NeuralSequantialDispatch(BaseDispatch):
    def __init__(self, encoder: GambleEncoder, network: BaseSequentialDispatchNetwork) -> None:
        super().__init__()
        self.encoder = encoder
        self.network = network

    def __call__(self, gamble: Gamble) -> Assignment:
        assignment_list = []
        gamble_embs_dict = self.encoder(gamble)
        courier_order_embs = gamble_embs_dict['couriers'] + gamble_embs_dict['orders']
        n_couriers = len(gamble.couriers)
        n_orders = len(gamble.orders)
        for claim_idx, claim_emb in enumerate(gamble_embs_dict['claims']):
            probas = self.network(claim_emb, courier_order_embs)
            choice = int(torch.argmax(probas).item())
            if choice == n_couriers + n_orders:
                continue
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
