import torch
import numpy as np
import logging

from src.dispatchs.base_dispatch import BaseDispatch
# from src.dispatchs.scorers import BaseScorer
from src.objects import (
    # Order,
    Gamble,
    Assignment,
)
from src.reinforcement.delivery import BaseActorCritic, DeliveryState
from src.utils import compulte_claims_to_couriers_distances

# logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.DEBUG,  filemode='w')
# LOGGER = logging.getLogger(__name__)


class NeuralSequantialDispatch(BaseDispatch):
    def __init__(self, actor_critic: BaseActorCritic, **kwargs) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.max_num_points_in_route = kwargs['max_num_points_in_route']
        self.use_dist = kwargs['use_dist']
        self.use_route = kwargs['use_route']

    def __call__(self, gamble: Gamble) -> Assignment:
        num_claims = len(gamble.claims)
        if num_claims == 0:
            return Assignment([])
        assignment_list = []
        available_couriers = gamble.couriers
        available_orders = gamble.orders
        prev_idxs: list[int] = []
        claims_to_couriers_distances = compulte_claims_to_couriers_distances(gamble)
        for claim_idx in range(num_claims):
            couriers_embs_list = [c.to_numpy() for c in available_couriers]
            orders_embs_list = [o.to_numpy(max_num_points_in_route=self.max_num_points_in_route, use_dist=self.use_dist,
                                           use_route=self.use_route) for o in available_orders]
            couriers_embs = np.stack(couriers_embs_list, axis=0) if len(couriers_embs_list) > 0 else None
            orders_embs = np.stack(orders_embs_list, axis=0) if len(orders_embs_list) > 0 else None
            orders_full_mask = [o.has_full_route(max_num_points_in_route=self.max_num_points_in_route)
                                for o in available_orders]

            state = DeliveryState(
                claim_emb=gamble.claims[claim_idx].to_numpy(use_dist=self.use_dist),
                couriers_embs=couriers_embs,
                orders_embs=orders_embs,
                prev_idxs=prev_idxs,
                orders_full_masks=orders_full_mask,
                claim_to_couries_dists=claims_to_couriers_distances[claim_idx],
                gamble_features=gamble.to_numpy(),
                claim_idx=claim_idx,
            )
            with torch.no_grad():
                self.actor_critic([state])
            assignment = self.actor_critic.get_actions_list(best_actions=True)[0].to_index()

            # ### DEBUG AREA
            # log_probs_chosen = self.actor_critic.get_log_probs_list()
            # log_probs = self.actor_critic.get_log_probs_tensor().exp()
            # len_c = len(state.couriers_embs) if state.couriers_embs is not None else 0
            # len_o = len(state.orders_embs) if state.orders_embs is not None else 0
            # LOGGER.debug(f'fake assignment: {assignment == len_c + len_o}, len_c: {len_c}, len_o: {len_o},
            # chosen probs: {log_probs_chosen}')
            # LOGGER.debug(str(log_probs))
            # ###

            if assignment < len(available_couriers):
                assignment_list.append((
                    available_couriers[assignment].id,
                    gamble.claims[claim_idx].id
                ))
                prev_idxs.append(assignment)
            elif assignment < len(available_couriers) + len(available_orders):
                assignment_list.append((
                    available_orders[assignment - len(available_couriers)].courier.id,
                    gamble.claims[claim_idx].id
                ))
                prev_idxs.append(assignment)
        return Assignment(assignment_list)


# def _make_is_full_mask(num_points_list: list[int], max_num_points_in_route: int) -> torch.FloatTensor:
#     is_full_mask = [n_points >= max_num_points_in_route - 1 for n_points in num_points_list] + [False]
#     # return torch.FloatTensor(is_full_mask) * -torch.inf
#     return torch.tensor(is_full_mask, dtype=torch.float) * -torch.inf
