import torch
from typing import Sequence, List
from torch.nn.utils.rnn import pad_sequence

from objects.gamble_triple import GambleTriple
from dispatch.solvers import HungarianSolver
from dispatch.scorings import ETAScoring

def create_mask(lenghts, device, mask_first)->torch.BoolTensor:
    '''
    returns True if element is masked, False otherwise
    '''
    max_len = max(lenghts)
    with torch.no_grad():
        assert max_len > 0, 'Tensor should not be empty'
        result = torch.arange(max_len, device=device).expand(len(lenghts), max_len) >= torch.tensor(lenghts, device=device).unsqueeze(1)
        if mask_first:
            result[:, 0] = True
        return result


def get_target_assignments(triple: GambleTriple, max_num_ords: int, max_num_crrs: int) -> List[int]:
    scorer = ETAScoring()
    solver = HungarianSolver()
    scores = scorer(triple.orders, triple.couriers)

    num_ords = len(triple.orders)

    assignments = [max_num_crrs] * num_ords + [-1] * (max_num_ords - num_ords)
    for o_idx, c_idx in zip(*solver(scores)):
        assignments[o_idx] = c_idx

    return assignments


def get_batch_embeddings_tensors(embeddings: List[torch.Tensor]):
    return {
        'o': pad_sequence([emb['o'] for emb in embeddings], batch_first=True, padding_value=0.0),
        'c': pad_sequence([emb['c'] for emb in embeddings], batch_first=True, padding_value=0.0),
        'ar': pad_sequence([emb['ar'] for emb in embeddings], batch_first=True, padding_value=0.0)
    }


def get_batch_masks(triples: List[GambleTriple], device):
    return {
        'o': pad_sequence([torch.BoolTensor([True] + [False] * len(triple.orders)) for triple in triples], batch_first=True, padding_value=True).to(device=device),
        'c': pad_sequence([torch.BoolTensor([True] + [False] * len(triple.couriers)) for triple in triples], batch_first=True, padding_value=True).to(device=device),
        'ar': pad_sequence([torch.BoolTensor([True] + [False] * len(triple.active_routes)) for triple in triples], batch_first=True, padding_value=True).to(device=device)
    }    


def get_cross_mask(masks) -> torch.FloatTensor:
    '''
    Input: a dict of torch-masks, where True corresponds to masked element
    Output: torch.Tensor of shape [bs, o, c + 1], where 0 corresponds to unmasked elements, -inf to masked ones
    '''
    with torch.no_grad():
        om_ones = torch.where(masks['o'], 0, 1).unsqueeze(-1).float()
        cm_ones = torch.where(masks['c'], 0, 1).unsqueeze(-2).float()
        inverse_binary_mask = torch.matmul(om_ones, cm_ones).float()

        real_part_mask = (1 - inverse_binary_mask) * -1e9 # [bs, o, c]
        fake_part_mask = torch.zeros((inverse_binary_mask.shape[0], inverse_binary_mask.shape[1], 1), device=inverse_binary_mask.device) # [bs, o, 1]

        return torch.cat([real_part_mask, fake_part_mask], dim=-1) # [bs, o, c + 1]
    

def cross_entropy_assignment_loss(pred_scores, target_assigments, cross_mask):
    '''
    Input: 
    * pred_scores - tensor of shape [bs, o + 1, c + 2]
    * target_scores - matrix of shape [bs, o]
    Output: loss
    '''
    assert len(pred_scores.shape) == 3, 'shape should be [bs, o, c+1]'
    assert pred_scores.shape[0] == len(target_assigments), 'batch size should be equal'
    assert pred_scores.shape[1] == len(target_assigments[0]) + 1, 'order dimention should differs one 1 (BOS item)'

    if pred_scores.shape[1] == 1: # no orders
        return 0
    
    # tgt_ass = torch.where(has_orders, mask.shape[2], -1)
    # for idx, assignments in enumerate(batch_assignments):
    #     for row, col in assignments:
    #         tgt_ass[idx][row + 1] = col + 1

    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    masked_scores_wo_bos = (cross_mask + pred_scores)[:, 1:, 1:]
    # print('masked_scores_wo_bos\n', masked_scores_wo_bos)
    # print('target_assigments\n', target_assigments)
    # print('#' * 50)
    logits = masked_scores_wo_bos.reshape(-1, masked_scores_wo_bos.shape[-1])
    classes = torch.tensor(target_assigments, device=pred_scores.device, dtype=torch.long).flatten()
    loss = ce_loss(logits, classes)
    return loss
    

def get_assignments_by_scores(pred_scores, masks, ids):
    '''
    Input:
    * pred_scores - tensor of shape [bs, o+1, c+2] with fake courier and BOS items
    * masks - a dict of tensors of shape [bs, o+1, c+1] with BOS items
    * ids - a sequence of dicts of ids without masked items and BOS items
    Output: a batch (sequence) of assignments (sequence of pairs)
    '''
    with torch.no_grad():
        fake_crr_idx = pred_scores.shape[-1] - 1
        assignments_batch = []
        argmaxes = torch.argmax(pred_scores, dim=-1).detach().cpu().numpy()
        for batch_idx in range(len(pred_scores)):
            assignments_batch.append([])
            assigned_orders = set()
            assigned_couriers = set()
            for o_idx, c_idx in enumerate(argmaxes[batch_idx]):
                if c_idx != fake_crr_idx \
                        and (not masks['o'][batch_idx][o_idx]) \
                        and (not masks['c'][batch_idx][c_idx]) \
                        and o_idx not in assigned_orders and c_idx not in assigned_couriers:
                    assignment = (ids[batch_idx]['o'][o_idx].item(), ids[batch_idx]['c'][c_idx].item())
                    assignments_batch[-1].append(assignment)
                    assigned_orders.add(o_idx)
                    assigned_couriers.add(c_idx)

        # print('pred_scores\n', pred_scores)
        # print('assignments_batch\n', assignments_batch)
        # print('#' * 50)
        return assignments_batch


