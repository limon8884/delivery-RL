import torch
import numpy as np

from src.dispatchs.base_dispatch import BaseDispatch
# from src.dispatchs.scorers import BaseScorer
from src.objects import (
    # Order,
    Gamble,
    Assignment,
)
from src.reinforcement.delivery import BaseActorCritic, DeliveryState


class NeuralSequantialDispatch(BaseDispatch):
    def __init__(self, actor_critic: BaseActorCritic, **kwargs) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.max_num_points_in_route = kwargs['max_num_points_in_route']

    def __call__(self, gamble: Gamble) -> Assignment:
        num_claims = len(gamble.claims)
        if num_claims == 0:
            return Assignment([])
        assignment_list = []
        available_couriers = gamble.couriers
        available_orders = gamble.orders
        prev_idxs = []
        for claim_idx in range(num_claims):
            couriers_embs_list = [c.to_numpy() for c in available_couriers]
            orders_embs_list = [o.to_numpy(max_num_points_in_route=self.max_num_points_in_route)
                                for o in available_orders]
            couriers_embs = np.stack(couriers_embs_list, axis=0) if len(couriers_embs_list) > 0 else None
            orders_embs = np.stack(orders_embs_list, axis=0) if len(orders_embs_list) > 0 else None

            state = DeliveryState(
                claim_emb=gamble.claims[claim_idx].to_numpy(),
                couriers_embs=couriers_embs,
                orders_embs=orders_embs,
                prev_idxs=prev_idxs
            )
            with torch.no_grad():
                self.actor_critic([state])
            assignment = self.actor_critic.get_actions_list(best_actions=True)[0].to_index()

            if assignment < len(available_couriers):
                assignment_list.append((
                    available_couriers[assignment].id,
                    gamble.claims[claim_idx].id
                ))
                prev_idxs.append(prev_idxs)
            elif assignment < len(available_couriers) + len(available_orders):
                assignment_list.append((
                    available_orders[assignment - len(available_couriers)].courier.id,
                    gamble.claims[claim_idx].id
                ))
                prev_idxs.append(prev_idxs)
        return Assignment(assignment_list)


def _make_is_full_mask(num_points_list: list[int], max_num_points_in_route: int) -> torch.FloatTensor:
    is_full_mask = [n_points >= max_num_points_in_route - 1 for n_points in num_points_list] + [False]
    # return torch.FloatTensor(is_full_mask) * -torch.inf
    return torch.tensor(is_full_mask, dtype=torch.float) * -torch.inf
