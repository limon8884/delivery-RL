import torch
import torch.nn as nn
from dispatch.solvers import HungarianSolver
from dispatch.scorings import ETAScoring
from collections import defaultdict

def make_target_score_tensor(np_scores, mask):
    with torch.no_grad():
        target = mask.float()
        for idx, scores in enumerate(np_scores):
            source = torch.tensor(scores, device=mask.device)
            target[idx][:source.shape[-2], :source.shape[-1]] = source
            
        return target

def make_target_assignment_tensor(np_scores, mask):
    solver = HungarianSolver()
    with torch.no_grad():
        target = torch.zeros(mask.shape, device=mask.device)
        for idx, scores in enumerate(np_scores):
            row_ids, col_ids = solver(scores)
            for row, col in zip(row_ids, col_ids):
                target[idx][row][col] = 1.0
            
        return target

def make_target_assignment_indexes_tensor(np_scores, mask):
    '''
    Shows to which courier the given order should be assigned
    '''
    solver = HungarianSolver()
    with torch.no_grad():
        tgt_ass = torch.where(mask[:, :, 0] == 1, mask.shape[2], -1)
        for idx, scores in enumerate(np_scores):
            row_ids, col_ids = solver(scores)
            for row, col in zip(row_ids, col_ids):
                tgt_ass[idx][row] = col

    return tgt_ass

def get_hard_assignments(preds):
    z = torch.zeros_like(preds)
    o = torch.ones_like(preds)
    preds_hard = torch.scatter(z, dim=-1, index=torch.argmax(preds, dim=-1, keepdim=True), src=o)

    return preds_hard

def get_loss(net, triples):
    scorer = ETAScoring()
    np_scores = [scorer(triple.orders, triple.couriers) for triple in triples]
    preds = net(triples, 0)
    mask = net.get_mask()
    tgt = make_target_score_tensor(np_scores, mask)

    mse_loss = nn.MSELoss(reduction='none')
    return (mse_loss(preds, tgt) * mask).sum() / mask.sum()

def masked_preds(preds, mask):
    mask_inf = (1 - mask) * -1e9
    mask_inf_add_fake = torch.cat([mask_inf, torch.zeros((mask.shape[0], mask.shape[1], 1), device=mask.device)], dim=-1)
    return mask_inf_add_fake + preds

def get_loss_solve(net, triples, metrics=defaultdict(list)):    
    scorer = ETAScoring()
    np_scores = [scorer(triple.orders, triple.couriers) for triple in triples]
    preds = net(triples, 0)
    mask = net.get_mask()
    tgt_ass = make_target_assignment_indexes_tensor(np_scores, mask)
    
    update_accuracy_metric(metrics, preds, tgt_ass)
    # update_grad_norm(metrics, net)

    ce_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    loss = ce_loss(masked_preds(preds, mask).transpose(1, 2), tgt_ass) / mask[:, :, 0].sum()
    return loss

def update_accuracy_metric(metrics, preds, tgt_ass):
    preds_hard = torch.argmax(preds, dim=-1)
    assert preds_hard.shape == tgt_ass.shape
    accuracy = ((preds_hard == tgt_ass) & (tgt_ass != -1)).sum() / (tgt_ass != -1).sum()
    metrics['accuracy'].append(accuracy.item())
    # eta_net = torch.max(preds, dim=-1).sum() / pred_ass.sum()
    # eta_solver = torch.max