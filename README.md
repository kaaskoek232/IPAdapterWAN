# ComfyUI-IPAdapter-WAN

This extension adapts the [InstantX IP-Adapter for SD3.5-Large](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) to work with **Wan 2.1** and other UNet-based video/image models in **ComfyUI**.

Unlike the original SD3 version (which depends on `joint_blocks` from MMDiT), this version performs **sampling-time identity conditioning** by dynamically injecting into **attention layers** — making it compatible with models like **Wan 2.1**, **AnimateDiff**, and other non-SD3 pipelines.

---

## 🚀 Features

- 🔁 Injects identity embeddings during sampling via attention block patching
- 🧠 Works with Wan 2.1 and other UNet-style models (no SD3/MMDiT required)
- 🛠️ Built on top of ComfyUI's IPAdapter framework
- 🎨 Enables consistent face/identity across frames in video workflows

---

## 📦 Installation

1. Clone the repo into your ComfyUI custom nodes directory:

```bash
git clone https://github.com/your-username/ComfyUI-IPAdapter-WAN.git
```

2. Download the required model weights:
- Download the IP-Adapter weights:
  
  - [`ip-adapter.bin`](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter/blob/main/ip-adapter.bin)
  
  - Place it in: `ComfyUI/models/ipadapter/`

- Download the CLIP Vision model:
  
  - [`siglip_vision_patch14_384.safetensors`](https://huggingface.co/Comfy-Org/sigclip_vision_384)
  
  - Place it in: `ComfyUI/models/clip_vision/`

*(Note: This model is based on [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384). The rehosted version yields nearly identical results.)*

---

## 🧠 How It Works

Wan models use a UNet structure instead of the DiT transformer blocks used in SD3. To make IPAdapter work with Wan:

- The extension scans all attention blocks (modules with `.to_q` and `.to_k`) dynamically.

- It injects IPAdapter's attention processors (`IPAttnProcessor`) directly into those blocks.

- Identity embeddings are updated based on the current sampling timestep using a learned resampler.

This means it works **without requiring joint_blocks or specific architectural assumptions** — making it plug-and-play for many custom models.

---

## 🛠 Usage

1. In ComfyUI, use the following nodes:
   
   - `Load IPAdapter WAN Model`
   
   - `Apply IPAdapter WAN Model`

2. Connect the `CLIP Vision` embedding (from a face image) and your diffusion model to the adapter.

3. Use a weight of **~0.5** as a good starting point.

4. You can apply this in video workflows to maintain consistent identity across frames.

---

## 📁 Example Workflows

Example `.json` workflows will be available soon in the `workflows/` folder.

---

## ✅ Compatibility

| Model            | Status              |
| ---------------- | ------------------- |
| Wan 2.1          | ✅ Works             |
| AnimateDiff      | ✅ Works             |
| SD3 / SDXL       | ❌ Use original repo |
| Any UNet variant | ✅ Likely to work    |

---

## 🔧 TODOs

- Allow multiple adapters without conflict

- Auto-detect model parameters (hidden size, num layers)

- Convert `.bin` to `safetensors` format

- Add more workflows for different models

---

## 🧑‍💻 Credits

- Adapted from: [InstantX IPAdapter for SD3.5](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter)

- ComfyUI extensions by: *your name / handle here*

---

Feel free to contribute or suggest improvements via GitHub Issues or Pull Requests.
