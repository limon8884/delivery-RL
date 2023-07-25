from typing import Any, List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from networks.encoders.positional_encoder import PositionalEncoder
from networks.encoders.point_encoder import PointEncoder
from networks.utils import create_mask
from objects.gamble_triple import GambleTriple


def unmask_BOS_items(masks):
    '''
    Operation applies NOT inplace
    '''
    masks_new = {
        'o': masks['o'].clone().detach(),
        'c': masks['c'].clone().detach(),
        'ar': masks['ar'].clone().detach()
    }
    masks_new['o'][..., 0] = False
    masks_new['c'][..., 0] = False
    masks_new['ar'][..., 0] = False

    return masks_new


class ScoringNet(nn.Module):
    def __init__(self,
                 mode='default',
                 n_layers=4,
                 d_model=128,
                 n_head=4,
                 dim_ff=128,
                 point_enc_dim=64,
                 number_enc_dim=8,
                 device=None,
                 dropout=0.1
                 ):
        super().__init__()
        self.mode = mode
        self.device = device
        self.number_enc_dim = number_enc_dim
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.n_layers = n_layers
        self.point_enc_dim = point_enc_dim

        self.encoders_AR = nn.ModuleDict({
            'o': nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_ff,
                batch_first=True,
                device=self.device,
                dropout=dropout
            ),
            'c': nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_ff,
                batch_first=True,
                device=self.device,
                dropout=dropout
            )
        })
        self.encoders_OC = nn.ModuleList([
            nn.ModuleDict({
                'o': nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=dim_ff,
                    batch_first=True,
                    device=self.device,
                    dropout=dropout
                ),
                'c': nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=dim_ff,
                    batch_first=True,
                    device=self.device,
                    dropout=dropout
                ),
            })
            for _ in range(n_layers)
        ])
        self.ord_fake_courier_head = nn.Sequential(
            nn.Linear(dim_ff, dim_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_ff, 1)
        ).to(device=device)
        self.ord_score_head = nn.Sequential(
            nn.Linear(dim_ff, dim_ff, device=device),
            nn.ReLU(),
            nn.Linear(dim_ff, 1, device=device),
            nn.Flatten(start_dim=-2)
        )

    def forward(self, tensors, masks):
        masks = unmask_BOS_items(masks)

        ord, crr = self.encoder_AR(tensors['o'], tensors['c'], tensors['ar'], masks)
        assert ord.isnan().sum() == 0 and crr.isnan().sum() == 0, 'nan values are not allowed\n' + str(ord)
        ord, crr = self.encoder_OC(ord, crr, masks)
        ord_scores, ord_fake_crr = self.score_and_fake_heads(ord)

        scores = self.bipartite_scores(ord, crr)
        final_scores = torch.cat([scores, ord_fake_crr], dim=-1)
        values = torch.mean(ord_scores, dim=-1)

        # unmask_BOS_items(masks)
        return final_scores, values

    def encoder_AR(self, ord, crr, ar, masks):
        new_ord = self.encoders_AR['o'](ord, ar, tgt_key_padding_mask=masks['o'], memory_key_padding_mask=masks['ar'])
        new_crr = self.encoders_AR['c'](crr, ar, tgt_key_padding_mask=masks['c'], memory_key_padding_mask=masks['ar'])

        return new_ord, new_crr

    def encoder_OC(self, ord, crr, masks):
        for encoders in self.encoders_OC:
            new_ord = encoders['o'](ord, crr, tgt_key_padding_mask=masks['o'], memory_key_padding_mask=masks['c'])
            new_crr = encoders['c'](crr, ord, tgt_key_padding_mask=masks['c'], memory_key_padding_mask=masks['o'])
            ord = new_ord
            crr = new_crr
        return ord, crr

    def bipartite_scores(self, ord, crr):
        crr_t = torch.transpose(crr, -2, -1)
        return torch.matmul(ord, crr_t)

    def score_and_fake_heads(self, ord):
        fake_head = self.ord_fake_courier_head(ord)  # [bs, o, 1]
        ord_scores = self.ord_score_head(ord)

        return ord_scores, fake_head


class ScoringInterface:
    def __init__(self, net: ScoringNet, point_encoder: PointEncoder = None) -> None:
        self.net = net
        self.device = self.net.device
        self.d_model = self.net.d_model

        self.order_enc = PositionalEncoder('o',
                                           point_encoder or PointEncoder(net.point_enc_dim, net.device),
                                           num_enc_dim=net.number_enc_dim,
                                           out_dim=net.d_model,
                                           device=net.device
                                           )
        self.courier_enc = PositionalEncoder('c',
                                             point_encoder or PointEncoder(net.point_enc_dim, net.device),
                                             num_enc_dim=net.number_enc_dim,
                                             out_dim=net.d_model,
                                             device=net.device
                                             )
        self.ar_enc = PositionalEncoder('ar',
                                        point_encoder or PointEncoder(net.point_enc_dim, net.device),
                                        num_enc_dim=net.number_enc_dim,
                                        out_dim=net.d_model,
                                        device=net.device
                                        )

    def load_weights(self, path_weights: str):
        self.load_encoder_weights(path_weights + '/encoders')
        self.load_net_weights(path_weights + '/net.pt')

    def encode_input(self, batch: List[GambleTriple] | GambleTriple, current_time):
        '''
        Attention: the function adds BOS-fake items to every GambleTriple to strugle zero-input problem
        '''
        unbatched_mode = isinstance(batch, GambleTriple)
        if unbatched_mode:
            batch = [batch]

        batch_c = []
        batch_o = []
        batch_ar = []
        for triple in batch:
            batch_o.append(triple.orders)
            batch_c.append(triple.couriers)
            batch_ar.append(triple.active_routes)

        o_tensor, o_mask, o_ids = self.make_tensor(batch_o, item_type='o',
                                                   current_time=current_time, unbatched_mode=unbatched_mode)
        c_tensor, c_mask, c_ids = self.make_tensor(batch_c, item_type='c',
                                                   current_time=current_time, unbatched_mode=unbatched_mode)
        ar_tensor, ar_mask, ar_ids = self.make_tensor(batch_ar, item_type='ar',
                                                      current_time=current_time, unbatched_mode=unbatched_mode)

        self.tensors = {
            'o': o_tensor,
            'c': c_tensor,
            'ar': ar_tensor
        }
        self.masks = {
            'o': o_mask,
            'c': c_mask,
            'ar': ar_mask
        }
        self.ids = {
            'o': o_ids,
            'c': c_ids,
            'ar': ar_ids
        }

    def make_tensor(self, batch_sequences, item_type, current_time, unbatched_mode):
        '''
        Input: a batch of sequences of items
        Output: an embedding tensor of shape [bs, max_len + 1, emb_size]
        where max_len - maximum sequence length over the batch; emb_size - d_model
        The function adds a BOS item with zeros-embedding
        '''
        samples = []
        lenghts = []
        ids_batch = []
        for sequence in batch_sequences:
            # if len(sequence) == 0:
            #     sample = torch.zeros((1, self.d_model), device=self.device, dtype=torch.float)
            #     ids = torch.tensor([-1], dtype=torch.int, device=self.device)
            bos_tensor = torch.zeros(self.d_model, device=self.device, dtype=torch.float)
            if item_type == 'o':
                sample = torch.stack([bos_tensor] + [self.order_enc(item, current_time) for item in sequence])
                ids = torch.tensor([-1] + [item.id for item in sequence], dtype=torch.int, device=self.device)
            elif item_type == 'c':
                sample = torch.stack([bos_tensor] + [self.courier_enc(item, current_time) for item in sequence])
                ids = torch.tensor([-1] + [item.id for item in sequence], dtype=torch.int, device=self.device)
            elif item_type == 'ar':
                sample = torch.stack([bos_tensor] + [self.ar_enc(item, current_time) for item in sequence])
                ids = torch.tensor([-1] + [item.id for item in sequence], dtype=torch.int, device=self.device)
            else:
                raise RuntimeError('unkown item type')
            samples.append(sample)
            lenghts.append(len(sequence) + 1)
            ids_batch.append(ids)

        tens = pad_sequence(samples, batch_first=True)
        masks = create_mask(lenghts, self.device, mask_first=True)
        ids_batch = pad_sequence(ids_batch, batch_first=True, padding_value=-1)

        if unbatched_mode:
            assert tens.shape[0] == 1, 'in unbatched mode should be 1'
            tens = tens.squeeze(dim=0)
            masks = masks.squeeze(dim=0)
            ids_batch = ids_batch.squeeze(dim=0)

        return tens, masks, ids_batch

    def inference(self) -> Any:
        assert self.tensors is not None, 'call encode first'
        self.scores, self.values = self.net(self.tensors, self.masks)

        return self.scores

    def get_assignments_batch(self, gamble_triples: List[GambleTriple]):
        assignments_batch = []
        with torch.no_grad():
            argmaxes = torch.argmax(self.scores, dim=-1)
            mask = self.get_mask()
            for batch_idx in range(len(self.scores)):
                assignments_batch.append([])
                assigned_orders = set()
                assigned_couriers = set()
                for o_idx, c_idx in enumerate(argmaxes[batch_idx].numpy()):
                    if c_idx != self.scores.shape[-1] - 1 and mask[batch_idx][o_idx][c_idx] \
                            and o_idx not in assigned_orders and c_idx not in assigned_couriers:
                        assignment = (
                            gamble_triples[batch_idx].orders[o_idx - 1].id,
                            gamble_triples[batch_idx].couriers[c_idx - 1].id
                            )
                        assignments_batch[-1].append(assignment)
                        assigned_orders.add(o_idx)
                        assigned_couriers.add(c_idx)
            return assignments_batch

    def CE_loss(self, batch_assignments):
        '''
        assignmets - is a batch of np.arrays arr, where arr[i] = j means that i-th order is assigned on j-th courier
        if there is no courier assigned, arr[i] = -1
        '''
        mask = self.get_mask()
        assert len(batch_assignments) == mask.shape[0]

        has_orders = mask.sum(dim=-1) > 0
        if has_orders.sum() == 0:
            return 0

        tgt_ass = torch.where(has_orders, mask.shape[2], -1)
        for idx, assignments in enumerate(batch_assignments):
            for row, col in assignments:
                tgt_ass[idx][row + 1] = col + 1

        mask_inf_add_fake = torch.cat(
            [(1 - mask) * -1e9, torch.zeros((mask.shape[0], mask.shape[1], 1), device=self.device)],
            dim=-1
            )

        ce_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        loss = ce_loss((mask_inf_add_fake + self.scores).transpose(1, 2), tgt_ass) / has_orders.sum()
        return loss

    def load_encoder_weights(self, path: str):
        self.order_enc.load_state_dict(torch.load(path + '/o.pt', map_location=self.device))
        self.courier_enc.load_state_dict(torch.load(path + '/c.pt', map_location=self.device))
        self.ar_enc.load_state_dict(torch.load(path + '/ar.pt', map_location=self.device))

    def load_net_weights(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def get_mask(self) -> torch.FloatTensor:
        '''
        Returns mask of shape [bs, num_orders, num_couriers + 1] (including BOS-items)
        1 is elements is not masked, 0 otherwise
        '''
        with torch.no_grad():
            om_ones = torch.where(self.masks['o'], 0, 1).unsqueeze(-1).float()
            cm_ones = torch.where(self.masks['c'], 0, 1).unsqueeze(-2).float()
            return torch.matmul(om_ones, cm_ones).float()
