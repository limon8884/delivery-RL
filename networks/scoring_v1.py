from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import *
from dispatch.utils import *
from networks.encoders import PositionalEncoder, PointEncoder
from networks.utils import *
from objects.gamble_triple import GambleTriple


class ScoringNet(nn.Module):
    def __init__(self, 
            mode='default', 
            # point_encoder=None,
            n_layers=4, 
            d_model=128, 
            n_head=4, 
            dim_ff=128, 
            point_enc_dim=64, 
            number_enc_dim=8,
            device=None,
            dropout=0.1):
        super().__init__()
        self.mode = mode
        self.device = device
        self.number_enc_dim = number_enc_dim
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.n_layers = n_layers
        self.point_enc_dim = point_enc_dim
        # self.point_encoder = point_encoder

        # self.order_enc = PositionalEncoder('o', 
        #                                        self.point_encoder, 
        #                                        num_enc_dim=number_enc_dim, 
        #                                        out_dim=d_model,  
        #                                        device=device
        #                                        )
        # self.courier_enc = PositionalEncoder('c', 
        #                                          self.point_encoder, 
        #                                          num_enc_dim=number_enc_dim, 
        #                                          out_dim=d_model, 
        #                                          device=device
        #                                          )
        # self.ar_enc = PositionalEncoder('ar', 
        #                                     self.point_encoder, 
        #                                     num_enc_dim=number_enc_dim, 
        #                                     out_dim=d_model, 
        #                                     device=device
        #                                 )     
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
        self.ord_score_head = nn.Linear(dim_ff, dim_ff, device=device)
    
    def forward(self, tensors, masks):
        # self.masks = None
        # ord, crr, ar = self.make_tensors_and_create_masks(batch, current_time)

        ord, crr = self.encoder_AR(tensors['o'], tensors['c'], tensors['ar'], masks)
        ord, crr = self.encoder_OC(ord, crr, masks)
        ord, ord_fake_crr = self.score_and_fake_heads(ord)

        scores = self.bipartite_scores(ord, crr)
        final_scores = torch.cat([scores, ord_fake_crr], dim=-1)

        return final_scores

    # def make_tensors_and_create_masks(self, batch: List[GambleTriple], current_time):
    #     batch_c = []
    #     batch_o = []
    #     batch_ar = []
    #     for triple in batch:
    #         batch_o.append(triple.orders)
    #         batch_c.append(triple.couriers)
    #         batch_ar.append(triple.active_routes)

    #     o_tensor, o_mask = self.make_tensor(batch_o, item_type='o', current_time=current_time)
    #     c_tensor, c_mask = self.make_tensor(batch_c, item_type='c', current_time=current_time)
    #     ar_tensor, ar_mask = self.make_tensor(batch_ar, item_type='ar', current_time=current_time)
    #     self.tensors = {
    #         'o': o_tensor,
    #         'c': c_tensor,
    #         'ar': ar_tensor
    #     }
    #     self.masks = {
    #         'o': o_mask,
    #         'c': c_mask,
    #         'ar': ar_mask
    #     }

    #     return o_tensor, c_tensor, ar_tensor

    # def make_tensor(self, batch_sequences, item_type, current_time):
    #     samples = []
    #     lenghts = []
    #     for sequence in batch_sequences:
    #         if item_type == 'o':
    #             sample = torch.stack([self.order_enc(item, current_time) for item in sequence])
    #         elif item_type == 'c':
    #             sample = torch.stack([self.courier_enc(item, current_time) for item in sequence])
    #         elif item_type == 'ar':
    #             sample = torch.stack([self.ar_enc(item, current_time) for item in sequence])
    #         samples.append(sample)
    #         lenghts.append(len(sequence))
        
    #     tens = pad_sequence(samples, batch_first=True)
    #     return tens, create_mask(lenghts, self.device)

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
        crr_t = torch.transpose(crr, 1, 2)
        return torch.matmul(ord, crr_t)

    def score_and_fake_heads(self, ord):
        fake_head = self.ord_fake_courier_head(ord) # [bs, o, 1]
        ord_scores = self.ord_score_head(ord)

        return ord_scores, fake_head

    # def get_mask(self):
    #     with torch.no_grad():
    #         om_ones = torch.where(self.masks['o'], 0, 1).unsqueeze(-1).float()
    #         cm_ones = torch.where(self.masks['c'], 0, 1).unsqueeze(-2).float()
    #         return torch.matmul(om_ones, cm_ones).float()


class ScoringInterface:
    def __init__(self, net: ScoringNet, point_encoder: PointEncoder = None) -> None:
        self.net = net
        self.device = self.net.device

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

    def encode_input(self, batch: List[GambleTriple], current_time):
        batch_c = []
        batch_o = []
        batch_ar = []
        for triple in batch:
            batch_o.append(triple.orders)
            batch_c.append(triple.couriers)
            batch_ar.append(triple.active_routes)

        o_tensor, o_mask = self.make_tensor(batch_o, item_type='o', current_time=current_time)
        c_tensor, c_mask = self.make_tensor(batch_c, item_type='c', current_time=current_time)
        ar_tensor, ar_mask = self.make_tensor(batch_ar, item_type='ar', current_time=current_time)
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

    def make_tensor(self, batch_sequences, item_type, current_time):
        samples = []
        lenghts = []
        for sequence in batch_sequences:
            if item_type == 'o':
                sample = torch.stack([self.order_enc(item, current_time) for item in sequence])
            elif item_type == 'c':
                sample = torch.stack([self.courier_enc(item, current_time) for item in sequence])
            elif item_type == 'ar':
                sample = torch.stack([self.ar_enc(item, current_time) for item in sequence])
            samples.append(sample)
            lenghts.append(len(sequence))
        
        tens = pad_sequence(samples, batch_first=True)
        return tens, create_mask(lenghts, self.device)
    
    def inference(self) -> Any:
        assert self.tensors is not None, 'call encode first'
        self.scores = self.net(self.tensors, self.masks)
    
        return self.scores   
    
    def CE_loss(self, batch_assignments):
        '''
        assignmets - is a batch of np.arrays arr, where arr[i] = j means that i-th order is assigned on j-th courier
        if there is no courier assigned, arr[i] = -1
        '''
        mask = self.get_mask()
        assert len(batch_assignments) == mask.shape[0]
    
        tgt_ass = torch.where(mask[:, :, 0] == 1, mask.shape[2], -1)
        for idx, assignments in enumerate(batch_assignments):
            # row_ids, col_ids = solver(scores)
            for row, col in assignments:
                tgt_ass[idx][row] = col

        mask_inf_add_fake = torch.cat([(1 - mask) * -1e9, torch.zeros((mask.shape[0], mask.shape[1], 1), device=self.device)], dim=-1)

        ce_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        loss = ce_loss((mask_inf_add_fake + self.scores).transpose(1, 2), tgt_ass) / mask[:, :, 0].sum()
        return loss

    def load_encoder_weights(self, path: str):
        self.order_enc.load_state_dict(torch.load(path + '/o.pt', map_location=self.device))
        self.courier_enc.load_state_dict(torch.load(path + '/c.pt', map_location=self.device))
        self.ar_enc.load_state_dict(torch.load(path + '/ar.pt', map_location=self.device))

    def load_net_weights(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def get_mask(self):
        with torch.no_grad():
            om_ones = torch.where(self.masks['o'], 0, 1).unsqueeze(-1).float()
            cm_ones = torch.where(self.masks['c'], 0, 1).unsqueeze(-2).float()
            return torch.matmul(om_ones, cm_ones).float()
        