""" 
HistoGPT Slide Encoder Module
© Manuel Tran / Helmholtz Munich
"""

import torch
import torch.nn as nn

from typing import List

from .embedder import NaViTEmbedding
from .normalization import FusedRMSNorm
from .perceiver import FlashPerceiver


class Aggregator(nn.Module):
    """
    The patch aggregation / slide encoder moduel.
    """
    def __init__(self, d_input: int = 1024, d_model: int = 1536, num_cls: int = 167):
        super().__init__()
        self.position = NaViTEmbedding(
            d_model=d_input,
            max_len=1024,
            patch_size=512,
        )
        self.model = FlashPerceiver(
            d_input=d_input,
            d_model=d_model,
            n_heads=16,
            n_layers=6,
            n_latents=640,
            attn_drop=0.0,
            concat_latents=True,
        )
        self.norm = FusedRMSNorm(
            normalized_shape=d_model,
            eps=1e-05,
            elementwise_affine=True,
        )
        self.head = nn.Linear(d_model, num_cls, False)

    def forward(self, x: List[torch.Tensor], pos: List[torch.Tensor]):
        x = [self.position(x[i], pos[i]) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        x = self.model(x)
        x = self.norm(x)
        x = x.unsqueeze(1) if x.ndim == 3 else x
        return x
