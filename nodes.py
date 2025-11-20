import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import folder_paths

try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("safetensors not available, falling back to torch.load")

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

# Cache for attention block names to avoid repeated iteration
# Key: model type name, Value: list of attention block names
_attention_block_cache: Dict[str, List[str]] = {}


def _get_attention_blocks(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """
    Cache attention block discovery for performance.
    Uses model type as cache key for stability.
    """
    model_type = type(model).__name__
    cache_key = f"{model_type}_{id(type(model))}"
    
    if cache_key not in _attention_block_cache:
        blocks = []
        for name, module in model.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_k"):
                blocks.append((name, module))
        _attention_block_cache[cache_key] = [name for name, _ in blocks]
        return blocks
    else:
        # Use cached names but still need to get modules
        blocks = []
        cached_names = _attention_block_cache[cache_key]
        module_dict = dict(model.named_modules())
        for name in cached_names:
            if name in module_dict:
                blocks.append((name, module_dict[name]))
        return blocks


def patch(
    patcher: Any,
    ip_procs: torch.nn.ModuleList,
    resampler: TimeResampler,
    clip_embeds: torch.Tensor,
    weight: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
) -> None:
    """
    Model-agnostic patcher that injects IPAdapter-like processors into any attention blocks.
    
    Optimized to cache attention block discovery and reduce redundant operations.
    """
    model = patcher.model.diffusion_model
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )

    # Use a shared dict for ip_options to avoid repeated dict lookups
    ip_options: Dict[str, Optional[torch.Tensor]] = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }

    def ddit_wrapper(forward: Any, args: Dict[str, Any]) -> torch.Tensor:
        """Optimized wrapper with reduced CPU-GPU transfers"""
        timestep_tensor = args["timestep"]
        # Avoid CPU transfer if possible - compute on GPU
        t_flattened = timestep_tensor.flatten()
        if t_flattened.numel() > 0:
            t_percent = 1.0 - t_flattened[0].item()
        else:
            t_percent = 0.0
            
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            cond_indices = args["cond_or_uncond"]
            embeds = clip_embeds[cond_indices]
            # Use expand + view instead of repeat_interleave when possible for better memory
            if batch_size > 1:
                embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            timestep = timestep_tensor * timestep_schedule_max
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], timestep_tensor, **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)

    # Use cached attention block discovery
    attention_blocks = _get_attention_blocks(model)
    n_procs = len(ip_procs)
    
    # Pre-allocate wrappers if possible
    for idx, (name, module) in enumerate(attention_blocks):
        wrapper = JointBlockIPWrapper(module, ip_procs[idx % n_procs], ip_options)
        patcher.set_model_patch_replace(wrapper, name)


def _load_checkpoint(checkpoint_path: str, device: str) -> Dict[str, Any]:
    """Load checkpoint with safetensors support for better performance and safety"""
    checkpoint_path_full = os.path.join(MODELS_DIR, checkpoint_path)
    
    if not os.path.exists(checkpoint_path_full):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_full}")
    
    # Try safetensors first if available
    if SAFETENSORS_AVAILABLE and checkpoint_path_full.endswith('.safetensors'):
        try:
            logging.info(f"Loading checkpoint using safetensors: {checkpoint_path_full}")
            return safetensors_load(checkpoint_path_full, device=device)
        except Exception as e:
            logging.warning(f"Failed to load safetensors, falling back to torch.load: {e}")
    
    # Fallback to torch.load
    logging.info(f"Loading checkpoint using torch.load: {checkpoint_path_full}")
    return torch.load(
        checkpoint_path_full,
        map_location=device,
        weights_only=True,
    )


class WANIPAdapter:
    def __init__(self, checkpoint: str, device: str):
        """
        Initialize WAN IP-Adapter with optimized loading and memory management.
        
        Args:
            checkpoint: Name of the checkpoint file
            device: Device to load the model on ('cuda', 'cpu', or 'mps')
        """
        self.device = device
        
        try:
            self.state_dict = _load_checkpoint(checkpoint, device)
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint}: {e}")
            raise
        
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
        
        # Move to device and set dtype before loading weights for efficiency
        self.resampler.to(device=device, dtype=torch.float16)
        
        if "image_proj" not in self.state_dict:
            raise KeyError("Missing 'image_proj' key in checkpoint")
        self.resampler.load_state_dict(self.state_dict["image_proj"])

        # Optimize processor count calculation
        if "ip_adapter" not in self.state_dict:
            raise KeyError("Missing 'ip_adapter' key in checkpoint")
        
        # Use set comprehension for faster unique extraction
        n_procs = len(set(x.split(".", 1)[0] for x in self.state_dict["ip_adapter"].keys()))
        
        # Pre-allocate processors in a single batch operation
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
        
        # Move all processors at once for better efficiency
        self.procs.to(device=device, dtype=torch.float16)
        self.procs.load_state_dict(self.state_dict["ip_adapter"])
        
        # Enable inference optimizations if available
        if hasattr(torch, 'compile') and device == 'cuda':
            try:
                self.resampler = torch.compile(self.resampler, mode='reduce-overhead')
                logging.info("Enabled torch.compile for resampler")
            except Exception as e:
                logging.warning(f"Could not compile resampler: {e}")


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
        end_percent: float,
    ) -> Tuple[Any]:
        """
        Apply IP-Adapter to the model with optimized tensor operations.
        
        Args:
            model: The diffusion model to apply IP-Adapter to
            ipadapter: The loaded IP-Adapter instance
            image_embed: CLIP vision output containing image embeddings
            weight: Strength of the IP-Adapter effect
            start_percent: Start timestep percentage (0.0-1.0)
            end_percent: End timestep percentage (0.0-1.0)
            
        Returns:
            Tuple containing the modified model
        """
        new_model = model.clone()
        image_embed_tensor = image_embed.penultimate_hidden_states
        
        # Optimize: pre-allocate on target device instead of cat then move
        device = ipadapter.device
        dtype = torch.float16
        
        # Create zero tensor directly on target device
        zero_embed = torch.zeros_like(image_embed_tensor, device=device, dtype=dtype)
        image_embed_on_device = image_embed_tensor.to(device=device, dtype=dtype)
        
        # Concatenate on device
        embeds = torch.cat([image_embed_on_device, zero_embed], dim=0)
        
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
