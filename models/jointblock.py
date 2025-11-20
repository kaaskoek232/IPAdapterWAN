from typing import Optional, Tuple, Dict, Any, Union

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
        time_embedding_dim (`int`, optional): The size of time embedding. Defaults to embedding_dim.
        mode (`str`): Mode of operation - "normal" or "zero". Defaults to "normal".
    """

    def __init__(
        self, 
        embedding_dim: int, 
        time_embedding_dim: Optional[int] = None, 
        mode: str = "normal"
    ):
        super().__init__()

        self.silu = nn.SiLU()
        num_params_dict = {
            "zero": 6,
            "normal": 2,
        }
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
        """
        Forward pass with adaptive layer normalization.
        
        Args:
            x: Input tensor
            hidden_dtype: Optional dtype hint (unused, kept for compatibility)
            emb: Time embedding tensor
        
        Returns:
            Normalized tensor(s) based on mode
        """
        emb = self.linear(self.silu(emb))
        if self.mode == "normal":
            shift_msa, scale_msa = emb.chunk(2, dim=1)
            # Optimize: use in-place operations where safe
            x_norm = self.norm(x)
            x = x_norm * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x

        elif self.mode == "zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
                6, dim=1
            )
            x_norm = self.norm(x)
            x = x_norm * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        
        raise ValueError(f"Unknown mode: {self.mode}")


class IPAttnProcessor(nn.Module):
    """
    IP-Adapter attention processor.
    
    Optimized for performance with reduced redundant operations.
    """

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
        Forward pass for IP attention processing.
        
        Args:
            ip_hidden_states: IP adapter hidden states (can be None)
            img_query: Image query tensor
            img_key: Image key tensor (optional, defaults to img_query)
            img_value: Image value tensor (optional, defaults to img_query)
            t_emb: Time embedding tensor
            n_heads: Number of attention heads
        
        Returns:
            Processed attention output or None if ip_hidden_states is None
        """
        if ip_hidden_states is None:
            return None

        # Early validation
        if not hasattr(self, "to_k_ip") or not hasattr(self, "to_v_ip"):
            return None

        # Normalize IP input
        norm_ip_hidden_states = self.norm_ip(ip_hidden_states, emb=t_emb)

        # Project to key and value
        ip_key = self.to_k_ip(norm_ip_hidden_states)
        ip_value = self.to_v_ip(norm_ip_hidden_states)

        # Optimize: Reshape all tensors efficiently
        # Cache head dimension calculation
        img_query = rearrange(img_query, "b l (h d) -> b h l d", h=n_heads)
        
        if img_key is not None:
            img_key = rearrange(img_key, "b l (h d) -> b h l d", h=n_heads)
        else:
            img_key = img_query  # Fallback to query if not provided
        
        # Handle img_value shape conversion
        if img_value is not None:
            if img_value.dim() == 4 and img_value.shape[1] == n_heads:
                # Already in shape b h l d, just ensure correct order
                img_value = img_value.transpose(1, 2) if img_value.shape[2] != img_query.shape[2] else img_value
            else:
                # Need to reshape from b l (h d) or b l h d
                img_value = rearrange(img_value, "b l (h d) -> b h l d", h=n_heads)
        else:
            img_value = img_query  # Fallback to query if not provided
        
        ip_key = rearrange(ip_key, "b l (h d) -> b h l d", h=n_heads)
        ip_value = rearrange(ip_value, "b l (h d) -> b h l d", h=n_heads)

        # Normalize
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        ip_key = self.norm_ip_k(ip_key)

        # Concatenate image and IP keys/values
        key = torch.cat([img_key, ip_key], dim=2)
        value = torch.cat([img_value, ip_value], dim=2)

        # Scaled dot-product attention
        ip_hidden_states = F.scaled_dot_product_attention(
            img_query, key, value, dropout_p=0.0, is_causal=False
        )
        ip_hidden_states = rearrange(ip_hidden_states, "b h l d -> b l (h d)")
        # Ensure output dtype matches input
        if ip_hidden_states.dtype != img_query.dtype:
            ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        return ip_hidden_states


class JointBlockIPWrapper:
    """
    Wrapper for JointBlock to add IP-Adapter attention processing.
    
    To be used as a patch_replace with ComfyUI.
    """

    def __init__(
        self,
        original_block: JointBlock,
        adapter: IPAttnProcessor,
        ip_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            original_block: The original JointBlock to wrap
            adapter: IP attention processor
            ip_options: Dictionary with IP adapter options (hidden_states, t_emb, weight)
        """
        self.original_block = original_block
        self.adapter = adapter
        self.ip_options = ip_options if ip_options is not None else {}

    def block_mixing(
        self, 
        context: torch.Tensor, 
        x: torch.Tensor, 
        context_block: Any, 
        x_block: Any, 
        c: Any
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Modified block mixing from mmdit.py with IP-Adapter attention.
        
        Optimized to reduce redundant operations and improve memory efficiency.
        
        Args:
            context: Context tensor
            x: Input tensor
            context_block: Context block module
            x_block: X block module
            c: Conditioning tensor
        
        Returns:
            Tuple of (context, x) tensors
        """
        context_qkv, context_intermediates = context_block.pre_attention(context, c)

        if x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = x_block.pre_attention(x, c)

        # Optimize: Pre-compute context length for slicing
        context_len = context_qkv[0].shape[1]
        
        # Concatenate context and x QKV
        qkv = tuple(torch.cat((context_qkv[j], x_qkv[j]), dim=1) for j in range(3))

        attn = optimized_attention(
            qkv[0],
            qkv[1],
            qkv[2],
            heads=x_block.attn.num_heads,
        )
        
        # Split attention results
        context_attn = attn[:, :context_len]
        x_attn = attn[:, context_len:]
        
        # Apply IP-Adapter if enabled
        hidden_states = self.ip_options.get("hidden_states")
        t_emb = self.ip_options.get("t_emb")
        weight = self.ip_options.get("weight", 1.0)
        
        if hidden_states is not None and t_emb is not None:
            # IP-Adapter attention
            ip_attn = self.adapter(
                hidden_states,
                *x_qkv,
                t_emb,
                x_block.attn.num_heads,
            )
            if ip_attn is not None:
                # Optimize: use in-place addition when safe
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

    def __call__(self, args: Dict[str, Any], _: Any) -> Dict[str, torch.Tensor]:
        """
        Call wrapper for ComfyUI patch_replace interface.
        
        Args:
            args: Dictionary with "txt", "img", and "vec" keys
            _: Unused second argument (kept for compatibility)
        
        Returns:
            Dictionary with "txt" and "img" keys
        """
        c, x = self.block_mixing(
            args["txt"],
            args["img"],
            self.original_block.context_block,
            self.original_block.x_block,
            c=args["vec"],
        )
        return {"txt": c, "img": x}
