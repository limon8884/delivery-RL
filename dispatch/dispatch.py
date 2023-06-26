from typing import List, Tuple
import itertools
import numpy as np

from utils import *
from dispatch.utils import *
from dispatch.generators import FullGenerator
from dispatch.scorings import ETAScoring
from dispatch.solvers import HungarianSolver
from objects.active_route import ActiveRoute
from objects.order import Order
from objects.courier import Courier
from objects.gamble_triple import GambleTriple

from networks.utils import get_batch_embeddings_tensors, get_batch_masks, get_assignments_by_scores

class BaseDispatch:
    '''
    Gets 3 lists as input:
    -free orders
    -free couriers
    -active routes
    returns pairs of indexes (order, courier) of first 2 lists - assigments
    '''
    def __init__(self) -> None:
        self.statistics = {
            "avg_scores": [],
            "num_assignments": []
        }

    def __call__(self, gamble_triple: GambleTriple) -> List[Tuple[int, int]]:
        pass


class Dispatch(BaseDispatch):
    def __init__(self) -> None:
        super().__init__()
        self.scoring = ETAScoring()
        self.solver = HungarianSolver()

    def __call__(self, gamble_triple: GambleTriple) -> List[Tuple[int, int]]:
        if len(gamble_triple.orders) == 0 or len(gamble_triple.couriers) == 0:
            return []
        
        scores = self.scoring(gamble_triple.orders, gamble_triple.couriers)
        assigned_order_idxs, assigned_courier_idxs = self.solver(scores)
        assignments = []
        for o_idx, c_idx in zip(assigned_order_idxs, assigned_courier_idxs):
            assignments.append((gamble_triple.orders[o_idx].id, gamble_triple.couriers[c_idx].id))

        # self.statistics['avg_scores'].append(np.mean([scores[ass[0], ass[1]] for ass in assignments]))
        self.statistics['num_assignments'].append(len(assignments))

        return assignments


# class NeuralDispatch(BaseDispatch):
#     def __init__(self, net) -> None:
#         super().__init__()
#         self.net = net
#         self.fallback_dispatch = Dispatch()

#     def __call__(self, gamble_triple: GambleTriple) -> List[Tuple[int, int]]:
#         if len(gamble_triple.orders) == 0 or len(gamble_triple.couriers) == 0:
#             return []
#         if len(gamble_triple.active_routes) == 0:
#             return self.fallback_dispatch(gamble_triple)
        
#         with torch.no_grad():
#             preds = self.net([gamble_triple], 0)[0]
#             fake_courier_idx = preds.shape[-1] - 1
#             argmaxes = torch.argmax(preds, dim=-1)

#             assignments = []
#             for o_idx, c_idx in enumerate(argmaxes):
#                 if c_idx < len(gamble_triple.couriers):
#                     assignments.append((o_idx, c_idx.item()))

#             self.statistics['num_assignments'].append(len(assignments))

#             return assignments

class NeuralDispatch(BaseDispatch):
    def __init__(self, net: nn.Module, encoder: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.net.eval()
        self.encoder = encoder
        self.encoder.eval()

    def __call__(self, batch_gamble_triples: List[GambleTriple]) -> List[List[Tuple[int, int]]]:
        with torch.no_grad():            
            embeds = []
            ids = []
            for triple in batch_gamble_triples:
                embeds_current, ids_current = self.encoder(triple, 0)
                embeds.append(embeds_current)
                ids.append(ids_current)

            batch_embs = get_batch_embeddings_tensors(embeds)
            batch_masks = get_batch_masks(batch_gamble_triples, device=self.net.device)

            pred_scores, _ = self.net(batch_embs, batch_masks)
            assignments_batch = get_assignments_by_scores(pred_scores, batch_masks, ids)

            return assignments_batch
        

