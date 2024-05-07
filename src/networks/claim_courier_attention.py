import torch
from torch import nn


class ClaimCourierAttention(nn.Module):
    def __init__(self,
                 clm_emb_size,
                 co_emb_size,
                 gmb_emb_size,
                 d_model,
                 nhead,
                 dim_feedforward,
                 num_attention_layers,
                 **kwargs,
                 ):
        super().__init__()
        self.device = kwargs['device']
        self.d_model = d_model
        self.clm_adaptor = nn.Linear(clm_emb_size, d_model).to(self.device)
        self.co_adaptor = nn.Linear(co_emb_size, d_model).to(self.device)
        self.gmb_adaptor = nn.Linear(gmb_emb_size, d_model).to(self.device)
        attention_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(attention_layer,
                                               num_layers=num_attention_layers).to(self.device)

    def forward(self, clm_embs: torch.Tensor, co_embs: torch.Tensor, gmb_emb: torch.Tensor, clm_masks: torch.Tensor,
                co_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = clm_embs.shape[0]
        clm_max_num = clm_embs.shape[1]
        co_max_num = co_embs.shape[1]
        assert clm_embs.shape == (bs, clm_max_num, self.clm_adaptor.in_features), clm_embs.shape
        assert co_embs.shape == (bs, co_max_num, self.co_adaptor.in_features), co_embs.shape
        assert gmb_emb.shape == (bs, self.gmb_adaptor.in_features), gmb_emb.shape
        co = self.co_adaptor(co_embs)
        clm = self.clm_adaptor(clm_embs)
        gmb = self.gmb_adaptor(gmb_emb).unsqueeze(1)
        gmb_masks = torch.tensor([False] * bs).unsqueeze(-1).to(self.device)
        inp = torch.cat([co, clm, gmb], dim=1)
        masks = torch.cat([co_masks, clm_masks, gmb_masks], dim=1)
        out = self.attention(inp, src_key_padding_mask=masks)
        assert out.shape == (bs, clm_max_num + co_max_num + 1, self.d_model), out.shape

        co_out = out[:, :co_max_num, :]
        assert co_out.shape == (bs, co_max_num, self.d_model)

        gmb_out = out[:, -1, :]
        assert gmb_out.shape == (bs, self.d_model)

        return co_out, gmb_out
