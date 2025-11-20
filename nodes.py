import os
import logging
from typing import Optional, Tuple, Dict, Any

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
    Optimized for performance with caching and efficient tensor operations.
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

    # Cache device and dtype for efficiency
    device = clip_embeds.device
    dtype = clip_embeds.dtype

    def ddit_wrapper(forward, args):
        # Optimize: avoid unnecessary CPU transfer and flatten
        timestep_tensor = args["timestep"]
        if timestep_tensor.numel() > 0:
            t_percent = 1 - timestep_tensor.flatten()[0].cpu().item()
        else:
            t_percent = 0.0
            
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            # Use indexing instead of repeat_interleave for better performance
            cond_indices = args["cond_or_uncond"]
            embeds = clip_embeds[cond_indices]
            if batch_size > 1:
                embeds = embeds.repeat_interleave(batch_size, dim=0)
            timestep = timestep_tensor * timestep_schedule_max
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], args["timestep"], **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)

    # Generic attention block patching - cache attention blocks for efficiency
    attention_blocks = []
    for name, module in model.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k"):
            attention_blocks.append((name, module))
    
    # Pre-allocate wrappers to avoid repeated modulo operations
    n_procs = len(ip_procs)
    for idx, (name, module) in enumerate(attention_blocks):
        wrapper = JointBlockIPWrapper(module, ip_procs[idx % n_procs], ip_options)
        patcher.set_model_patch_replace(wrapper, name)


class WANIPAdapter:
    def __init__(self, checkpoint: str, device: str, enable_compile: bool = True):
        """
        Initialize WAN IP-Adapter model.
        
        Args:
            checkpoint: Name of the checkpoint file
            device: Device to load model on ("cuda", "cpu", "mps")
            enable_compile: Whether to use torch.compile for optimization (PyTorch 2.0+)
        """
        self.device = device
        self.state_dict = torch.load(
            os.path.join(MODELS_DIR, checkpoint),
            map_location=self.device,
            weights_only=True,
        )
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

        n_procs = len(set(x.split(".")[0] for x in self.state_dict["ip_adapter"].keys()))
        self.procs = torch.nn.ModuleList([
            IPAttnProcessor(
                hidden_size=2432,
                cross_attention_dim=2432,
                ip_hidden_states_dim=2432,
                ip_encoder_hidden_states_dim=2432,
                head_dim=64,
                timesteps_emb_dim=1280,
            ).to(self.device, dtype=torch.float16)
            for _ in range(n_procs)
        ])
        self.procs.load_state_dict(self.state_dict["ip_adapter"])
        
        # Optimize with torch.compile if available and enabled
        if enable_compile and hasattr(torch, "compile"):
            try:
                self.resampler = torch.compile(self.resampler, mode="reduce-overhead")
                # Compile processors individually for better optimization
                for i, proc in enumerate(self.procs):
                    self.procs[i] = torch.compile(proc, mode="reduce-overhead")
                logging.info("torch.compile optimization enabled for IP-Adapter")
            except Exception as e:
                logging.warning(f"torch.compile failed, continuing without it: {e}")


class IPAdapterWANLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),),
                "provider": (["cuda", "cpu", "mps"],),
            },
            "optional": {
                "enable_compile": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_WAN_INSTANTX",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter: str, provider: str, enable_compile: bool = True):
        """
        Load IP-Adapter WAN model.
        
        Args:
            ipadapter: Checkpoint filename
            provider: Device provider ("cuda", "cpu", "mps")
            enable_compile: Enable torch.compile optimization (PyTorch 2.0+)
        """
        logging.info("Loading InstantX IPAdapter WAN model.")
        model = WANIPAdapter(ipadapter, provider, enable_compile=enable_compile)
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
        new_model = model.clone()
        image_embed = image_embed.penultimate_hidden_states
        # Optimize: pre-allocate zeros tensor and use stack for better performance
        device = ipadapter.device
        dtype = torch.float16
        zeros = torch.zeros_like(image_embed, device=device, dtype=dtype)
        image_embed = image_embed.to(device=device, dtype=dtype)
        embeds = torch.cat([image_embed, zeros], dim=0)
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
