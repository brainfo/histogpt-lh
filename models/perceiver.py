""" 
(Flash) Perceiver Resampler
© Jaegle et al. / Google DeepMind
"""

import torch
import torch.nn as nn

from flash_perceiver import Perceiver


class FlashPerceiver(nn.Module):
    """
    The Perceiver model with FlashAttention.
    """
    def __init__(
        self,
        d_input: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        n_latents: int = 256,
        attn_drop: float = 0.0,
        concat_latents: bool = True,
    ):
        super().__init__()
        self.concat_latents = concat_latents
        if concat_latents:
            self.latents = nn.Parameter(torch.randn(n_latents, d_model))
        else:
            self.latents = None  # use the model's learned latent vectors

        self.linear = nn.Linear(d_input, d_model, bias=False)

        self.perceiver = Perceiver(
            input_dim=d_model,
            depth=n_layers,
            output_dim=None,
            output_mode='average',
            num_latents=n_latents,
            latent_dim=d_model,
            cross_heads=n_heads if concat_latents else 1,
            cross_head_dim=d_model // n_heads,
            cross_rotary_emb_dim=0,
            cross_attn_dropout=attn_drop,
            latent_heads=n_heads,
            latent_head_dim=d_model // n_heads,
            latent_rotary_emb_dim=0,
            latent_attn_dropout=attn_drop,
            weight_tie_layers=False,
            gated_mlp=True,
            self_per_cross_attn=0 if concat_latents else 1,
            num_zero_tokens=None,
            use_flash_attn=True,
        )

    def forward(self, x: torch.Tensor):
        b, _, _ = x.size()
        x = self.linear(x)
        if self.concat_latents:
            l = self.latents.repeat(b, 1, 1)
            x = torch.cat((x, l), dim=1)
        else:
            l = None
        return self.perceiver(data=x, latents=l)
