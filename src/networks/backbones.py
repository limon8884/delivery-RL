import torch
from torch import nn


class TransformerBackbone(nn.Module):
    def __init__(self,
                 claim_embedding_dim,
                 courier_order_embedding_dim,
                 nhead,
                 hidden_dim,
                 dim_feedforward,
                 num_encoder_layers,
                 num_decoder_layers,
                 **kwargs) -> None:
        super().__init__()
        self.clm_emb_dim = claim_embedding_dim
        self.co_emb_dim = courier_order_embedding_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        device = kwargs['device']

        # claim encoder
        claim_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.clm_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.claim_encoder_block = nn.TransformerEncoder(encoder_layer=claim_encoder_layer,
                                                         num_layers=num_encoder_layers).to(device)
        self.claim_fc = nn.Linear(self.clm_emb_dim, self.hidden_dim + self.co_emb_dim).to(device)

        # courier-order decoder
        courier_order_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.co_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.courier_order_decoder_block = nn.TransformerDecoder(decoder_layer=courier_order_decoder_layer,
                                                                 num_layers=num_decoder_layers).to(device)
        self.courier_order_fc = nn.Linear(self.co_emb_dim, 2 * self.hidden_dim).to(device)

        # PPO heads
        # self.policy_head = nn.Sequential(
        #     nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 1)
        # ).to(device)
        self.value_head = nn.Linear(self.hidden_dim, 1).to(device)

    def forward(self, co_embs: torch.FloatTensor, clm_embs: torch.FloatTensor, co_masks: torch.FloatTensor,
                clm_padding_mask: torch.BoolTensor, co_padding_mask: torch.BoolTensor
                ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        bs = len(co_embs)
        assert co_embs.size(0) == bs and co_embs.size(2) == self.co_emb_dim, co_embs.shape
        assert clm_embs.size(0) == bs and clm_embs.size(2) == self.clm_emb_dim, clm_embs.shape
        assert (co_masks.size(0) == bs * self.nhead) and (co_masks.size(1) == co_masks.size(2)), co_masks.shape
        assert co_masks.dtype == torch.float, co_masks.dtype
        max_clm_len = clm_embs.size(1)
        max_co_len = co_embs.size(1)
        assert clm_padding_mask.size(0) == bs and clm_padding_mask.size(1) == max_clm_len, clm_padding_mask.shape
        assert clm_padding_mask.dtype == torch.bool, clm_padding_mask.dtype
        assert co_padding_mask.size(0) == bs and co_padding_mask.size(1) == max_co_len, co_padding_mask.shape
        assert co_padding_mask.dtype == torch.bool, co_padding_mask.dtype

        encoded_clms = self.claim_encoder_block(clm_embs, src_key_padding_mask=clm_padding_mask)
        encoded_clms = self.claim_fc(encoded_clms)
        policy_encoded_clms, memory_encoded_clms = torch.split(
            encoded_clms,
            split_size_or_sections=[self.hidden_dim, self.co_emb_dim],
            dim=-1
        )
        encoded_cocs = self.courier_order_decoder_block(tgt=co_embs, memory=memory_encoded_clms,
                                                        tgt_mask=co_masks,
                                                        tgt_key_padding_mask=co_padding_mask,
                                                        memory_key_padding_mask=clm_padding_mask)
        encoded_cocs = self.courier_order_fc(encoded_cocs)
        policy_encoded_cocs, value_encoded_cocs = torch.split(
            encoded_cocs,
            split_size_or_sections=[self.hidden_dim, self.hidden_dim],
            dim=-1
        )

        value = self.value_head(value_encoded_cocs[:, 0, :]).squeeze(-1)
        assert value.ndim == 1, value.ndim
        assert value.size(0) == bs, value.shape

        policy = policy_encoded_clms @ policy_encoded_cocs.transpose(-1, -2)
        assert policy.ndim == 3, policy.ndim
        assert policy.size(0) == bs and policy.size(1) == max_clm_len and policy.size(2) == max_co_len, policy.shape

        return policy, value
