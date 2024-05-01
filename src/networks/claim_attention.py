import torch
from torch import nn


class ClaimAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs['device']
        self.clm_emb_dim = kwargs['claim_embedding_dim']
        self.fc = nn.Linear(self.clm_emb_dim + 1, self.clm_emb_dim).to(self.device)
        attention_layer = nn.TransformerEncoderLayer(
            d_model=self.clm_emb_dim,
            nhead=kwargs['nhead'],
            dim_feedforward=kwargs['dim_feedforward'],
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(attention_layer,
                                               num_layers=kwargs['num_attention_layers']).to(self.device)
        # self.idx_embed = nn.Embedding(3, embedding_dim=kwargs['idx_embedding_dim'])

    def forward(self, clm_embs: torch.Tensor, claim_idx: int) -> torch.Tensor:
        n_clms = clm_embs.shape[0]
        assert clm_embs.shape == (n_clms, self.clm_emb_dim), clm_embs.shape
        idxs = torch.where(torch.arange(n_clms).to(self.device) > claim_idx, 1.0, 0.0)
        idxs[claim_idx] = -1.0
        x = torch.cat([clm_embs, idxs.unsqueeze(-1)], dim=-1)
        x = self.fc(x)
        x = self.attention(x)
        x = x.mean(dim=0).unsqueeze(0)
        assert x.shape == (1, self.clm_emb_dim), x.shape
        # assert attn.shape[1] == self.clm_emb_dim, clm_embs.shape
        return x
