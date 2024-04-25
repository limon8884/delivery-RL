import torch
from torch import nn


class ClaimAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        device = kwargs['device']
        self.clm_emb_dim = kwargs['claim_embedding_dim']
        attention_layer = nn.TransformerEncoderLayer(
            d_model=self.clm_emb_dim,
            nhead=kwargs['nhead'],
            dim_feedforward=kwargs['dim_feedforward']
        )
        self.attention = nn.TransformerEncoder(attention_layer, num_layers=kwargs['num_attention_layers']).to(device)

    def forward(self, clm_embs: torch.Tensor) -> torch.Tensor:
        assert clm_embs.shape[1] == self.clm_emb_dim, clm_embs.shape
        attn = self.attention(clm_embs)
        # x = torch.cat([clm_embs, attn], dim=-1)
        assert attn.shape[1] == self.clm_emb_dim, clm_embs.shape
        return attn
