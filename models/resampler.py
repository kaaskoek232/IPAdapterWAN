# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn

from diffusers.models.embeddings import Timesteps, TimestepEmbedding


# FFN
def FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    """
    Feed-forward network module.
    
    Args:
        dim: Input/output dimension
        mult: Multiplier for inner dimension
    
    Returns:
        Sequential module with LayerNorm, Linear, GELU, Linear
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x: torch.Tensor, heads: int) -> torch.Tensor:
    """
    Optimized tensor reshaping for attention heads.
    
    Args:
        x: Input tensor of shape (bs, length, width)
        heads: Number of attention heads
    
    Returns:
        Reshaped tensor of shape (bs, heads, length, dim_per_head)
    """
    bs, length, width = x.shape
    dim_per_head = width // heads
    
    # Optimize: Use view + transpose instead of multiple reshape operations
    # (bs, length, width) --> (bs, length, heads, dim_per_head)
    x = x.view(bs, length, heads, dim_per_head)
    # (bs, length, heads, dim_per_head) --> (bs, heads, length, dim_per_head)
    x = x.transpose(1, 2).contiguous()
    return x


class PerceiverAttention(nn.Module):
    """
    Perceiver attention module optimized for performance.
    """
    
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
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

    def forward(
        self, 
        x: torch.Tensor, 
        latents: torch.Tensor, 
        shift: Optional[torch.Tensor] = None, 
        scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Perceiver attention.
        
        Args:
            x: Image features of shape (b, n1, D)
            latents: Latent features of shape (b, n2, D)
            shift: Optional shift tensor for adaptive normalization
            scale: Optional scale tensor for adaptive normalization
        
        Returns:
            Output tensor of shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        if shift is not None and scale is not None:
            # Optimize: use in-place operations where safe
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Optimize: Pre-compute scale factor
        attn_scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # Attention computation - more stable with f16
        weight = (q * attn_scale) @ (k * attn_scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).to(weight.dtype)
        out = weight @ v

        # Optimize: Use contiguous reshape
        out = out.permute(0, 2, 1, 3).contiguous().reshape(b, l, -1)

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
    """
    Time-aware resampler for IP-Adapter embeddings.
    
    Optimized for memory efficiency and performance.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        embedding_dim: int = 768,
        output_dim: int = 1024,
        ff_mult: int = 4,
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
    ):
        super().__init__()

        # Initialize learnable latents
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        # Build layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # Multi-head self-attention
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # Feed-forward network
                        FeedForward(dim=dim, mult=ff_mult),
                        # Adaptive layer norm modulation
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
                    ]
                )
            )

        # Time embedding components
        self.time_proj = Timesteps(
            timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift
        )
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")

    def forward(
        self, 
        x: torch.Tensor, 
        timestep: torch.Tensor, 
        need_temb: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through time-aware resampler.
        
        Args:
            x: Input embeddings of shape (bs, seq_len, embedding_dim)
            timestep: Timestep tensor
            need_temb: Whether to return time embedding
        
        Returns:
            Resampled latents, optionally with time embedding
        """
        timestep_emb = self.embedding_time(x, timestep)  # bs, dim

        # Optimize: Use expand instead of repeat when possible
        batch_size = x.size(0)
        latents = self.latents.expand(batch_size, -1, -1)

        x = self.proj_in(x)
        # Add time embedding
        x = x + timestep_emb.unsqueeze(1)

        # Process through layers
        for attn, ff, adaLN_modulation in self.layers:
            # Get adaptive normalization parameters
            adaLN_params = adaLN_modulation(timestep_emb)
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_params.chunk(4, dim=1)
            
            # Attention with residual
            latents = attn(x, latents, shift_msa, scale_msa) + latents

            # Feed-forward with adaptive layer norm
            res = latents
            for idx_ff, layer_ff in enumerate(ff):
                latents = layer_ff(latents)
                # Apply adaptive normalization after first LayerNorm
                if idx_ff == 0 and isinstance(layer_ff, nn.LayerNorm):
                    latents = latents * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            
            # Residual connection
            latents = latents + res

        # Final projection and normalization
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        if need_temb:
            return latents, timestep_emb
        else:
            return latents

    def embedding_time(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int]
    ) -> torch.Tensor:
        """
        Generate time embeddings from timestep.
        
        Optimized to reduce CPU-GPU synchronization.
        
        Args:
            sample: Sample tensor (for device/dtype inference)
            timestep: Timestep value(s)
        
        Returns:
            Time embedding tensor
        """
        # Convert timestep to tensor if needed
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep

        # Broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])

        # Project timesteps
        t_emb = self.time_proj(timesteps)

        # Cast to match sample dtype (important for fp16)
        t_emb = t_emb.to(dtype=sample.dtype)

        # Generate embedding
        emb = self.time_embedding(t_emb, None)
        return emb
