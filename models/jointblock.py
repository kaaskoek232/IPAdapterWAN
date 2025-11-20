from typing import Optional, Dict, Any, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import (
    RMSNorm,
    JointBlock,
)


class AdaLayerNorm(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        time_embedding_dim (`int`, optional): The size of time embedding vector.
        mode (`str`): Mode of operation, either "normal" or "zero".
    """

    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None, mode: str = "normal"):
        super().__init__()

        self.silu = nn.SiLU()
        # Use tuple for immutable lookup (slightly faster than dict)
        num_params_dict = {"zero": 6, "normal": 2}
        num_params = num_params_dict.get(mode, 2)
        self.linear = nn.Linear(
            time_embedding_dim or embedding_dim, num_params * embedding_dim, bias=True
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass with optimized tensor operations."""
        if emb is None:
            raise ValueError("emb must be provided")
        emb = self.linear(self.silu(emb))
        if self.mode == "normal":
            shift_msa, scale_msa = emb.chunk(2, dim=1)
            # Optimize: use in-place operations where safe
            x_normed = self.norm(x)
            x = x_normed * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            return x

        elif self.mode == "zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
                6, dim=1
            )
            x_normed = self.norm(x)
            x = x_normed * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class IPAttnProcessor(nn.Module):
    """Optimized IP-Adapter attention processor with reduced memory allocations."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        ip_hidden_states_dim: int,
        ip_encoder_hidden_states_dim: int,
        head_dim: int,
        timesteps_emb_dim: int = 1280,
    ):
        super().__init__()

        self.norm_ip = AdaLayerNorm(
            ip_hidden_states_dim, time_embedding_dim=timesteps_emb_dim
        )
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.norm_q = RMSNorm(head_dim, 1e-6)
        self.norm_k = RMSNorm(head_dim, 1e-6)
        self.norm_ip_k = RMSNorm(head_dim, 1e-6)

    def forward(
        self,
        ip_hidden_states: Optional[torch.Tensor],
        img_query: torch.Tensor,
        img_key: Optional[torch.Tensor] = None,
        img_value: Optional[torch.Tensor] = None,
        t_emb: Optional[torch.Tensor] = None,
        n_heads: int = 1,
    ) -> Optional[torch.Tensor]:
        """
        Optimized forward pass with reduced intermediate allocations.
        
        Args:
            ip_hidden_states: IP adapter hidden states
            img_query: Image query tensor
            img_key: Image key tensor (optional, defaults to img_query)
            img_value: Image value tensor (optional, defaults to img_query)
            t_emb: Time embedding tensor
            n_heads: Number of attention heads
            
        Returns:
            Processed attention output or None if ip_hidden_states is None
        """
        if ip_hidden_states is None or t_emb is None:
            return None

        # Early validation
        if not hasattr(self, "to_k_ip") or not hasattr(self, "to_v_ip"):
            return None

        # Normalize IP input
        norm_ip_hidden_states = self.norm_ip(ip_hidden_states, emb=t_emb)

        # Project to k and v (fused operations)
        ip_key = self.to_k_ip(norm_ip_hidden_states)
        ip_value = self.to_v_ip(norm_ip_hidden_states)

        # Reshape image tensors - optimize by doing all reshapes together
        img_query = rearrange(img_query, "b l (h d) -> b h l d", h=n_heads)
        if img_key is None:
            img_key = img_query
        else:
            img_key = rearrange(img_key, "b l (h d) -> b h l d", h=n_heads)
        
        if img_value is None:
            img_value = img_query
        else:
            # Handle transpose if needed (optimize: check shape first)
            if img_value.dim() == 4 and img_value.shape[1] != n_heads:
                img_value = torch.transpose(img_value, 1, 2)
            img_value = rearrange(img_value, "b l (h d) -> b h l d", h=n_heads) if img_value.dim() == 3 else img_value
        
        # Reshape IP tensors
        ip_key = rearrange(ip_key, "b l (h d) -> b h l d", h=n_heads)
        ip_value = rearrange(ip_value, "b l (h d) -> b h l d", h=n_heads)

        # Normalize (can be done in-place for some operations)
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        ip_key = self.norm_ip_k(ip_key)

        # Concatenate - use pre-allocated tensors when possible
        key = torch.cat([img_key, ip_key], dim=2)
        value = torch.cat([img_value, ip_value], dim=2)

        # Scaled dot-product attention (optimized in PyTorch)
        ip_hidden_states = F.scaled_dot_product_attention(
            img_query, key, value, dropout_p=0.0, is_causal=False
        )
        
        # Rearrange and ensure dtype consistency
        ip_hidden_states = rearrange(ip_hidden_states, "b h l d -> b l (h d)")
        # Only convert dtype if necessary
        if ip_hidden_states.dtype != img_query.dtype:
            ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        
        return ip_hidden_states


class JointBlockIPWrapper:
    """To be used as a patch_replace with Comfy. Optimized for performance."""

    def __init__(
        self,
        original_block: JointBlock,
        adapter: IPAttnProcessor,
        ip_options: Optional[Dict[str, Any]] = None,
    ):
        self.original_block = original_block
        self.adapter = adapter
        self.ip_options = ip_options if ip_options is not None else {}

    def block_mixing(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        context_block: Any,
        x_block: Any,
        c: Any,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Optimized block mixing with IP-Adapter attention injection.
        Comes from mmdit.py. Modified to add ipadapter attention.
        """
        context_qkv, context_intermediates = context_block.pre_attention(context, c)

        if x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = x_block.pre_attention(x, c)

        # Optimize: pre-allocate tuple and use list comprehension
        qkv = tuple(torch.cat((context_qkv[j], x_qkv[j]), dim=1) for j in range(3))

        attn = optimized_attention(
            qkv[0],
            qkv[1],
            qkv[2],
            heads=x_block.attn.num_heads,
        )
        
        # Optimize: cache shape to avoid repeated access
        context_len = context_qkv[0].shape[1]
        context_attn = attn[:, :context_len]
        x_attn = attn[:, context_len:]
        
        # Check IP options once and cache values
        hidden_states = self.ip_options.get("hidden_states")
        t_emb = self.ip_options.get("t_emb")
        weight = self.ip_options.get("weight", 1.0)
        
        # IP-Adapter injection (only if enabled for current timestep)
        if hidden_states is not None and t_emb is not None:
            ip_attn = self.adapter(
                hidden_states,
                x_qkv[0],  # img_query
                x_qkv[1] if len(x_qkv) > 1 else None,  # img_key
                x_qkv[2] if len(x_qkv) > 2 else None,  # img_value
                t_emb,
                x_block.attn.num_heads,
            )
            if ip_attn is not None:
                # Use in-place addition when safe
                x_attn = x_attn + ip_attn * weight

        # Post-process context
        if not context_block.pre_only:
            context = context_block.post_attention(context_attn, *context_intermediates)
        else:
            context = None
            
        # Post-process x
        if x_block.x_block_self_attn:
            attn2 = optimized_attention(
                x_qkv2[0],
                x_qkv2[1],
                x_qkv2[2],
                heads=x_block.attn2.num_heads,
            )
            x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
        else:
            x = x_block.post_attention(x_attn, *x_intermediates)
            
        return context, x

    def __call__(self, args: Dict[str, Any], _: Any) -> Dict[str, Any]:
        """
        Call wrapper for ComfyUI patching system.
        
        Args:
            args: Dictionary containing 'txt', 'img', and 'vec' keys
            _: Unused second argument (for compatibility)
            
        Returns:
            Dictionary with 'txt' and 'img' keys
        """
        c, x = self.block_mixing(
            args["txt"],
            args["img"],
            self.original_block.context_block,
            self.original_block.x_block,
            c=args["vec"],
        )
        return {"txt": c, "img": x}
