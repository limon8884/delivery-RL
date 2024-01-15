import torch
import torch.nn as nn


class BaseSequentialDispatchNetwork(nn.Module):
    def forward(self, claim_embedding: torch.FloatTensor,
                courier_order_embeddings: list[torch.FloatTensor]) -> torch.FloatTensor:
        raise NotImplementedError


class SimpleSequentialMLP(BaseSequentialDispatchNetwork):
    def __init__(self, claim_embedding_dim, courier_order_embedding_dim, device) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(claim_embedding_dim, courier_order_embedding_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(courier_order_embedding_dim, courier_order_embedding_dim, device=device),
            nn.LeakyReLU(),
        )
        self.fake_courier_embedding = torch.zeros(size=(courier_order_embedding_dim,),
                                                  dtype=torch.float32, device=device)

    def forward(self, claim_embedding: torch.FloatTensor,
                courier_order_embeddings: list[torch.FloatTensor]) -> torch.FloatTensor:
        x = self.mlp(claim_embedding)
        y = torch.stack(courier_order_embeddings + [self.fake_courier_embedding], dim=-2)
        z = x @ y.transpose(-1, -2)
        assert z.shape[-1] == len(courier_order_embeddings) + 1
        return nn.functional.softmax(z, dim=-1)
