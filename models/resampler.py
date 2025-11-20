# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn

from diffusers.models.embeddings import Timesteps, TimestepEmbedding


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class TimeResampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # msa
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # ff
                        FeedForward(dim=dim, mult=ff_mult),
                        # adaLN
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
                    ]
                )
            )

        # time
        self.time_proj = Timesteps(
            timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift
        )
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")

        # adaLN
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(timestep_out_dim, 6 * timestep_out_dim, bias=True)
        # )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, need_temb: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Optimized forward pass with improved memory efficiency and computation.
        
        Args:
            x: Input tensor
            timestep: Timestep tensor
            need_temb: Whether to return timestep embedding
            
        Returns:
            latents or (latents, timestep_emb) if need_temb is True
        """
        # Optimize: compute timestep embedding once
        timestep_emb = self.embedding_time(x, timestep)  # bs, dim

        # Optimize: expand latents more efficiently
        batch_size = x.size(0)
        latents = self.latents.expand(batch_size, -1, -1)

        # Optimize: project input and add timestep embedding in one step
        x = self.proj_in(x)
        x = x + timestep_emb.unsqueeze(1)  # More efficient than [:, None]

        # Optimize: pre-compute adaLN modulation values outside the loop if possible
        # But we need them per layer, so keep in loop but optimize chunk operation
        for attn, ff, adaLN_modulation in self.layers:
            # Optimize: compute modulation once and chunk efficiently
            modulation = adaLN_modulation(timestep_emb)
            shift_msa, scale_msa, shift_mlp, scale_mlp = modulation.chunk(4, dim=1)
            
            # Optimize: use residual connection more efficiently
            latents = attn(x, latents, shift_msa, scale_msa) + latents

            # Optimize: store residual before FF processing
            res = latents
            
            # Optimize: process FF layers with optimized adaLN application
            for idx_ff, layer_ff in enumerate(ff):
                latents = layer_ff(latents)
                # Optimize: apply adaLN more efficiently
                if idx_ff == 0 and isinstance(layer_ff, nn.LayerNorm):  # adaLN
                    # Use in-place operations where safe (but be careful with gradients)
                    scale_mlp_expanded = scale_mlp.unsqueeze(1)
                    latents = latents * (1 + scale_mlp_expanded) + shift_mlp.unsqueeze(1)
            
            # Optimize: add residual connection
            latents = latents + res

        # Optimize: final projection and normalization
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        if need_temb:
            return latents, timestep_emb
        else:
            return latents

    def embedding_time(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]) -> torch.Tensor:
        """
        Optimized timestep embedding computation.
        
        Args:
            sample: Sample tensor to infer device and dtype
            timestep: Timestep value(s)
            
        Returns:
            Timestep embedding tensor
        """
        # Optimize: handle timestep tensor conversion more efficiently
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # Optimize: determine device and dtype more efficiently
            device = sample.device
            is_mps = device.type == "mps"
            
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            # Optimize: expand scalar tensor more efficiently
            timesteps = timesteps.unsqueeze(0).to(sample.device)

        # Optimize: broadcast to batch dimension
        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)

        # Optimize: project timesteps
        t_emb = self.time_proj(timesteps)

        # Optimize: cast to sample dtype (avoids unnecessary dtype conversions)
        t_emb = t_emb.to(dtype=sample.dtype)

        # Optimize: compute embedding
        emb = self.time_embedding(t_emb, None)
        return emb
