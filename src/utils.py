import torch
import torch.nn as nn
import json
from src.dispatch.solvers import HungarianSolver
from src.dispatch.scorings import ETAScoring
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import Counter
from src.objects.gamble_triple import random_triple
from src.objects.point import Point
from tqdm import tqdm
import typing
from typing import Sequence, Dict, Any, Callable
from joblib import Parallel, delayed


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
    mask_inf_add_fake = torch.cat(
        [mask_inf, torch.zeros((mask.shape[0], mask.shape[1], 1), device=mask.device)],
        dim=-1
        )
    return mask_inf_add_fake + preds


def get_loss_solve(model, triples, metrics=defaultdict(list)):
    scorer = ETAScoring()
    np_scores = [scorer(triple.orders, triple.couriers) for triple in triples]
    model.encode_input(triples, 0)
    preds = model.inference()
    mask = model.get_mask()
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


def add_avg_grad_norm_metric(metrics, net):
    grad_norms = []
    for name, param in net.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad_norms.append(torch.square(param.grad).mean().item())
    metrics['grad_norm'].append(np.log(np.mean(grad_norms)))


def print_info(epoch, metrics, losses):
    clear_output()
    print('EPOCH: ', epoch)
    print(losses[-50:])
    # plt.plot(np.log(losses))
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title('loss')
    plt.plot(losses)
    plt.subplot(1, 3, 2)
    plt.title('accuracy')
    plt.plot(metrics['accuracy'])
    plt.subplot(1, 3, 3)
    plt.title('grad_norm')
    plt.plot(metrics['grad_norm'])
    plt.show()


def target_assigments(triple):
    scoring = ETAScoring()
    solver = HungarianSolver()

    scores = scoring(triple.orders, triple.couriers)
    ords, crrs = solver(scores)
    answer = -np.ones(len(triple.orders))
    for o, c in zip(ords, crrs):
        answer[o] = c
    return answer


def pred_assigments(net, triple):
    preds = net([triple], 0)
    fake_courier_idx = preds[0].shape[1] - 1
    argmaxes = torch.argmax(preds[0], dim=-1)
    return torch.where(argmaxes == fake_courier_idx, -1, argmaxes).numpy()


def target_assigments_batch(triples):
    assignments_batch = []
    for triple in triples:
        assignments_batch.append(target_assigments(triple))

    return assignments_batch


def pred_assigments_batch(net, triples):
    preds = net(triples, 0)
    fake_courier_idx = preds.shape[2] - 1
    argmaxes = torch.argmax(preds, dim=-1)
    assignments_batch_tens = torch.where(argmaxes == fake_courier_idx, -1, argmaxes).numpy()

    assignments_batch = []
    for i, assignments_tens in enumerate(assignments_batch_tens):
        num_orders = len(triples[i].orders)
        assignments_batch.append(assignments_tens[:num_orders])

    return assignments_batch


def get_dict_metrics(a, b, n_couriers=1000):
    return {
        'fake_mistake_not_assigns': ((a != b) & (a == -1) & (b < n_couriers)).sum(),
        'fake_mistake_assigns': ((a != b) & (b == -1)).sum(),
        'real_mistakes': ((a != b) & (a != -1) & (b != -1) & (b < n_couriers)).sum(),
        'masked_assigns': ((b >= n_couriers)).sum(),
        'correct': (a == b).sum(),
    }


def get_metrics(net, n_samples, max_items=3, bounds=(Point(0, 0), Point(10, 10))):
    c = Counter()
    for _ in tqdm(range(n_samples)):
        triple = random_triple(bounds, max_items=max_items)
        pred_ass = pred_assigments(net, triple)
        tgt_ass = target_assigments(triple)
        assert pred_ass.shape == tgt_ass.shape
        c.update(get_dict_metrics(tgt_ass, pred_ass))

    return c


def get_metrics_batch(net, batch_size, n_samples, max_items=3, bounds=(Point(0, 0), Point(10, 10))):
    c = Counter()
    for _ in tqdm(range(n_samples)):
        triples = [random_triple(bounds, max_items=max_items) for _ in range(batch_size)]
        pred_ass_batch = pred_assigments_batch(net, triples)
        tgt_ass_batch = target_assigments_batch(triples)
        assert len(pred_ass_batch) == len(tgt_ass_batch)
        for i, (pred_ass, tgt_ass) in enumerate(zip(pred_ass_batch, tgt_ass_batch)):
            assert pred_ass.shape == tgt_ass.shape
            c.update(get_dict_metrics(tgt_ass, pred_ass, n_couriers=len(triples[i].couriers)))

    return c


def get_batch_quality_metrics(dispatch: typing.Any, simulator_type: typing.Any, batch_size: int, num_steps: int):
    '''
    Runs given dispatch in simulator and collects downstream metrics
    '''
    simulators = [simulator_type() for i in range(batch_size)]
    all_metrics = [[] for _ in range(batch_size)]

    for step in range(num_steps):
        triples = [sim.GetState() for sim in simulators]
        assignments = dispatch(triples)
        for i in range(batch_size):
            simulators[i].Next(assignments[i])
            all_metrics[i].append(simulators[i].GetMetrics())

    return all_metrics


def get_CR(batch_metrics):
    '''
    Takes downstream metrics and returns total CR
    '''
    num = sum([metric[-1]['completed_orders'] for metric in batch_metrics])
    denom = sum([metric[-1]['finished_orders'] for metric in batch_metrics])
    if denom == 0:
        return 0
    return num / denom


def update_assignment_accuracy_statistics(tgt: typing.List[int], pred: torch.Tensor, statistics: typing.Counter):
    '''
    Takes a gamble correct assignments and predictions and returns the quality statistics
    Input:
    * tgt - list of order assignments of length ord
    * pred - tensor of shape [ord+1, crr+2]
    '''
    preds_np = pred[1:, 1:].argmax(dim=-1).cpu().numpy()  # [ord]
    tgts_np = np.array(tgt)
    n_couriers = pred.shape[-1] - 2
    n_real_couriers = (tgts_np != -1).sum() - 1

    statistics.update({
        'fake_mistake_not_assigns': ((tgts_np == n_couriers) & (preds_np != n_couriers)).sum(),
        'fake_mistake_assigns': ((tgts_np != n_couriers) & (tgts_np != -1) & (preds_np == n_couriers)).sum(),
        'masked_assigns': ((tgts_np == -1) & ((preds_np < n_real_couriers) | (preds_np == n_couriers))).sum(),
        'real_mistakes': ((tgts_np != preds_np) & (tgts_np != n_couriers)
                          & (preds_np != n_couriers) & (tgts_np != -1)).sum(),
        'not_masked_couriers': (tgts_np != -1).sum(),
        'correct': ((tgts_np == preds_np) & (tgts_np != -1)).sum(),
        }
    )


def update_run_counters(mode='test'):
    with open('configs/run_ids.json') as f:
        data = json.load(f)
    data[mode] += 1
    new_json = json.dumps(data)
    with open('configs/run_ids.json', 'w') as f:
        f.write(new_json)


def aggregate_metrics(data: Sequence[Dict[str, Any]], agg_func: Callable):
    result = {}
    for key in data[0]:
        result[key] = agg_func([d[key] for d in data])
    return result
