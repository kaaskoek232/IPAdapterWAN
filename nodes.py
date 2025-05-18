import os
import logging

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
    Model-agnostic patcher that injects IPAdapter-like processors into any attention blocks
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

    def ddit_wrapper(forward, args):
        t_percent = 1 - args["timestep"].flatten()[0].cpu().item()
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            embeds = clip_embeds[args["cond_or_uncond"]]
            embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            timestep = args["timestep"] * timestep_schedule_max
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], args["timestep"], **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)

    # Generic attention block patching
    idx = 0
    for name, module in model.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k"):
            wrapper = JointBlockIPWrapper(module, ip_procs[idx % len(ip_procs)], ip_options)
            patcher.set_model_patch_replace(wrapper, name)
            idx += 1


class WANIPAdapter:
    def __init__(self, checkpoint: str, device):
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
        new_model = model.clone()
        image_embed = image_embed.penultimate_hidden_states
        embeds = torch.cat([image_embed, torch.zeros_like(image_embed)], dim=0).to(
            ipadapter.device, dtype=torch.float16
        )
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
