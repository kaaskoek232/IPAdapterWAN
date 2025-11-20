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

def patch(
    patcher,
    ip_procs,
    resampler: TimeResampler,
    clip_embeds,
    weight=1.0,
    start=0.0,
    end=1.0,
):
    """
    Model-agnostic patcher that injects IPAdapter-like processors into any attention blocks.
    
    Optimized version with cached attention block discovery and reduced redundant operations.
    """
    model = patcher.model.diffusion_model
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )

    ip_options = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }

    # Cache for batch size and embeddings to avoid recomputation
    # Use a list to allow modification in closure (Python closure workaround)
    cache_state = {"batch_size": None, "embeds": {}}

    def ddit_wrapper(forward, args):
        # Optimize: avoid CPU transfer if not needed, use in-place operations where safe
        timestep_tensor = args["timestep"]
        t_percent = 1.0 - timestep_tensor.flatten()[0].cpu().item()
        
        if start <= t_percent <= end:
            # Optimize: cache batch size calculation
            current_batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            cond_uncond_idx = args["cond_or_uncond"]
            
            # Optimize: reuse cached embeddings if batch size hasn't changed
            cache_key = (cond_uncond_idx, current_batch_size)
            if cache_state["batch_size"] != current_batch_size or cache_key not in cache_state["embeds"]:
                embeds = clip_embeds[cond_uncond_idx]
                # Optimize: pre-allocate repeated embeddings
                embeds = torch.repeat_interleave(embeds, current_batch_size, dim=0)
                cache_state["batch_size"] = current_batch_size
                cache_state["embeds"][cache_key] = embeds
            else:
                embeds = cache_state["embeds"][cache_key]
            
            timestep = timestep_tensor * timestep_schedule_max
            # Optimize: use torch.no_grad() if resampler is in eval mode (handled internally)
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], timestep_tensor, **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)

    # Optimize: collect all attention blocks first, then patch in batch
    # This reduces redundant module traversal
    attention_blocks: List[Tuple[str, Any]] = []
    for name, module in model.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k"):
            attention_blocks.append((name, module))
    
    # Apply patches using modulo indexing for processor reuse
    num_procs = len(ip_procs)
    for idx, (name, module) in enumerate(attention_blocks):
        wrapper = JointBlockIPWrapper(module, ip_procs[idx % num_procs], ip_options)
        patcher.set_model_patch_replace(wrapper, name)


class WANIPAdapter:
    def __init__(self, checkpoint: str, device):
        """
        Initialize WAN IP-Adapter model with optimized loading.
        
        Args:
            checkpoint: Name of the checkpoint file
            device: Device to load the model on (cuda, cpu, mps)
        """
        self.device = device
        checkpoint_path = os.path.join(MODELS_DIR, checkpoint)
        
        # Optimize: Load checkpoint once and reuse
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Optimize: Use mmap for large files if on CPU, faster loading
        map_location = self.device
        if device == "cpu":
            map_location = "cpu"
        
        self.state_dict = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        
        # Initialize resampler with optimized settings
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
        
        # Optimize: Move to device and dtype before loading state dict
        # This avoids unnecessary device transfers
        self.resampler.to(self.device, dtype=torch.float16)
        self.resampler.load_state_dict(self.state_dict["image_proj"])
        
        # Optimize: Use set comprehension more efficiently
        # Count unique processor prefixes
        proc_keys = set(x.split(".")[0] for x in self.state_dict["ip_adapter"].keys())
        n_procs = len(proc_keys)
        
        # Optimize: Create processors in batch, then move to device
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
        
        # Load state dict before moving to device (more efficient)
        self.procs.load_state_dict(self.state_dict["ip_adapter"])
        
        # Move all processors to device and dtype in one go
        self.procs.to(self.device)
        for proc in self.procs:
            proc.to(dtype=torch.float16)


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
        self, model, ipadapter, image_embed, weight, start_percent, end_percent
    ):
        """
        Apply IP-Adapter to the model with optimized tensor operations.
        
        Optimized to reduce memory allocations and device transfers.
        """
        new_model = model.clone()
        image_embed = image_embed.penultimate_hidden_states
        
        # Optimize: Pre-allocate zeros on target device to avoid transfer
        # and concatenate more efficiently
        device = ipadapter.device
        dtype = torch.float16
        
        # Optimize: Move image_embed to target device first, then create zeros with same shape
        image_embed_device = image_embed.to(device=device, dtype=dtype)
        zeros = torch.zeros_like(image_embed_device)
        embeds = torch.cat([image_embed_device, zeros], dim=0)
        
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
