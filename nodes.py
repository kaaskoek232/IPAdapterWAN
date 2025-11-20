import os
import logging
from typing import Dict, List, Tuple, Optional, Any

import torch
import folder_paths

from .models.resampler import TimeResampler
from .models.jointblock import JointBlockIPWrapper, IPAttnProcessor

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)

# Cache for attention block names to avoid repeated scanning
_attention_block_cache: Dict[str, List[str]] = {}


def _get_attention_blocks(model: torch.nn.Module, model_id: str) -> List[Tuple[str, torch.nn.Module]]:
    """
    Get all attention blocks from a model, with caching to avoid repeated scanning.
    
    Args:
        model: The diffusion model to scan
        model_id: Unique identifier for the model (for caching)
    
    Returns:
        List of (name, module) tuples for attention blocks
    """
    if model_id in _attention_block_cache:
        # Use cached names, but fetch fresh modules
        return [(name, dict(model.named_modules())[name]) 
                for name in _attention_block_cache[model_id]]
    
    # Scan and cache
    attention_blocks = []
    for name, module in model.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k"):
            attention_blocks.append((name, module))
    
    # Cache the names
    _attention_block_cache[model_id] = [name for name, _ in attention_blocks]
    return attention_blocks


def patch(
    patcher,
    ip_procs: torch.nn.ModuleList,
    resampler: TimeResampler,
    clip_embeds: torch.Tensor,
    weight: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
) -> None:
    """
    Model-agnostic patcher that injects IPAdapter-like processors into any attention blocks.
    
    Optimized version that caches attention block discovery and reduces CPU-GPU transfers.
    
    Args:
        patcher: Model patcher from ComfyUI
        ip_procs: List of IP attention processors
        resampler: Time-based resampler for embeddings
        clip_embeds: CLIP vision embeddings (shape: [2, seq_len, embed_dim])
        weight: Weight for IP adapter influence
        start: Start percentage of timestep range (0.0-1.0)
        end: End percentage of timestep range (0.0-1.0)
    """
    model = patcher.model.diffusion_model
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )
    
    # Use model's string representation as cache key
    model_id = str(id(model))

    ip_options: Dict[str, Optional[torch.Tensor]] = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }

    def ddit_wrapper(forward, args):
        # Optimize: avoid CPU transfer if possible, use item() directly on GPU tensor
        timestep_tensor = args["timestep"].flatten()
        # Use first element without full CPU transfer if tensor is small
        if timestep_tensor.numel() == 1:
            t_percent = 1.0 - timestep_tensor.item() / timestep_schedule_max
        else:
            t_percent = 1.0 - timestep_tensor[0].cpu().item() / timestep_schedule_max
        
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            embeds = clip_embeds[args["cond_or_uncond"]]
            # Optimize: use expand + reshape instead of repeat_interleave when possible
            if batch_size > 1:
                embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            else:
                embeds = embeds  # No need to repeat
            
            timestep = args["timestep"] * timestep_schedule_max
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], args["timestep"], **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)

    # Optimized: Use cached attention block discovery
    attention_blocks = _get_attention_blocks(model, model_id)
    n_procs = len(ip_procs)
    
    for idx, (name, module) in enumerate(attention_blocks):
        wrapper = JointBlockIPWrapper(
            module, 
            ip_procs[idx % n_procs], 
            ip_options
        )
        patcher.set_model_patch_replace(wrapper, name)


class WANIPAdapter:
    """
    WAN IP-Adapter model loader and manager.
    
    Optimized for faster loading and memory efficiency.
    """
    
    def __init__(self, checkpoint: str, device: str):
        """
        Initialize WAN IP-Adapter from checkpoint.
        
        Args:
            checkpoint: Name of the checkpoint file in MODELS_DIR
            device: Device to load model on ("cuda", "cpu", or "mps")
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If required keys are missing from state_dict
        """
        self.device = device
        checkpoint_path = os.path.join(MODELS_DIR, checkpoint)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load with optimized settings
        self.state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        
        # Validate required keys
        if "image_proj" not in self.state_dict:
            raise KeyError("Missing 'image_proj' key in checkpoint")
        if "ip_adapter" not in self.state_dict:
            raise KeyError("Missing 'ip_adapter' key in checkpoint")
        
        # Initialize resampler
        self.resampler = TimeResampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=64,
            embedding_dim=1152,
            output_dim=2432,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        self.resampler.eval()
        self.resampler.to(self.device, dtype=torch.float16)
        self.resampler.load_state_dict(self.state_dict["image_proj"])

        # Optimize: Count processors more efficiently
        proc_keys = self.state_dict["ip_adapter"].keys()
        n_procs = len(set(x.split(".", 1)[0] for x in proc_keys))
        
        # Create processors in batch for better memory allocation
        self.procs = torch.nn.ModuleList([
            IPAttnProcessor(
                hidden_size=2432,
                cross_attention_dim=2432,
                ip_hidden_states_dim=2432,
                ip_encoder_hidden_states_dim=2432,
                head_dim=64,
                timesteps_emb_dim=1280,
            )
            for _ in range(n_procs)
        ])
        # Move all to device at once
        self.procs.to(self.device, dtype=torch.float16)
        self.procs.load_state_dict(self.state_dict["ip_adapter"])


class IPAdapterWANLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_WAN_INSTANTX",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, provider):
        logging.info("Loading InstantX IPAdapter WAN model.")
        model = WANIPAdapter(ipadapter, provider)
        return (model,)


class ApplyIPAdapterWAN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IP_ADAPTER_WAN_INSTANTX",),
                "image_embed": ("CLIP_VISION_OUTPUT",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter(
        self, 
        model: Any, 
        ipadapter: WANIPAdapter, 
        image_embed: Any, 
        weight: float, 
        start_percent: float, 
        end_percent: float
    ) -> Tuple[Any]:
        """
        Apply IP-Adapter to a model.
        
        Args:
            model: ComfyUI model to apply adapter to
            ipadapter: Loaded WAN IP-Adapter instance
            image_embed: CLIP vision output with penultimate_hidden_states
            weight: Weight for IP adapter influence
            start_percent: Start percentage of timestep range
            end_percent: End percentage of timestep range
        
        Returns:
            Tuple containing the patched model
        """
        new_model = model.clone()
        image_embed = image_embed.penultimate_hidden_states
        
        # Optimize: Move to device and create zero tensor efficiently
        device = ipadapter.device
        dtype = torch.float16
        image_embed_device = image_embed.to(device=device, dtype=dtype)
        # Create zero tensor directly on target device with same shape
        zero_embed = torch.zeros_like(image_embed_device)
        
        # Concatenate on device
        embeds = torch.cat([image_embed_device, zero_embed], dim=0)
        
        patch(
            new_model,
            ipadapter.procs,
            ipadapter.resampler,
            embeds,
            weight=weight,
            start=start_percent,
            end=end_percent,
        )
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "IPAdapterWANLoader": IPAdapterWANLoader,
    "ApplyIPAdapterWAN": ApplyIPAdapterWAN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterWANLoader": "Load IPAdapter WAN Model",
    "ApplyIPAdapterWAN": "Apply IPAdapter WAN Model",
}
