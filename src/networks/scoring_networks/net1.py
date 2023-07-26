import torch
import torch.nn as nn


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
                 n_layers,
                 d_model,
                 n_head,
                 dim_ff,
                 device=None,
                 path_weights=None,
                 dropout=0.1):
        super().__init__()
        self.device = device
        self.d_model = d_model  # item emb size
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.n_layers = n_layers

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
        self.ord_scores_head = nn.Sequential(
            nn.Linear(dim_ff, dim_ff, device=device),
            nn.ReLU(),
            nn.Linear(dim_ff, 1, device=device),
            nn.Flatten(start_dim=-2)
        )

        if path_weights is not None:
            self.load_weights(path_weights)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print('net weights loaded successfuly!')

    def forward(self, tensors, masks):
        '''
        Input: tensors of shape [bs, o, c, emb] and masks
        Returns scores matrixes of shape [bs, o, c+1] and values of gambles of shape [bs]
        '''
        masks = unmask_BOS_items(masks)

        ord, crr = self.encoder_AR(tensors['o'], tensors['c'], tensors['ar'], masks)
        assert ord.isnan().sum() == 0 and crr.isnan().sum() == 0, 'nan values are not allowed\n' + str(ord)
        ord, crr = self.encoder_OC(ord, crr, masks)
        ord_fake_crr = self.fake_crr_scores(ord)

        scores = self.bipartite_scores(ord, crr)
        final_scores = torch.cat([scores, ord_fake_crr], dim=-1)
        values = self.ord_values(ord, masks['o'])

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

    def fake_crr_scores(self, ord):
        fake_head = self.ord_fake_courier_head(ord)  # [bs, o, 1]
        return fake_head

    def state_value(self, ord, ord_mask):
        '''
        Input: orders [bs, o, emb], ord_masks [bs, o]
        '''
        ord_scores = self.ord_scores_head(ord)  # [bs, o]
        numerator = torch.sum(torch.where(ord_mask, 0.0, ord_scores.double()), dim=-1)
        denominator = torch.sum(torch.where(ord_mask, 0.0, 1.0), dim=-1)

        return numerator / denominator  # [bs]

    def ord_values(self, ord, ord_mask):
        '''
        Input: orders [bs, o, emb], ord_masks [bs, o]
        '''
        ord_scores = self.ord_scores_head(ord)  # [bs, o]
        return torch.where(ord_mask, torch.tensor(0, dtype=torch.float, device=self.device), ord_scores)
