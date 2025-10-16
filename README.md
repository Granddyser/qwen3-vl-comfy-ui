# Qwen3-VL for ComfyUI

## 🆕 Update Log

### v1.0.2 (2025-10-16)
- Added **Qwen3_VL_NSFW_Run** node  
  → NSFW captioning and analysis support  
  → Category: `Qwen3-VL_NSFW`

---

This project is **inspired by and based on prior work** from [makki-shizu](https://github.com/MakkiShizu/), who originally implemented the **Qwen2.5-VL** nodes.  
All code has been updated, rewritten, and adapted for **Qwen3-VL** models.

---

## Description

This custom ComfyUI node set adds native support for **Qwen3-VL** models — the latest multimodal models from Alibaba Cloud's Qwen team.

Supported input types:
- Text
- Image
- Multi-image
- Video

### Key Differences from Qwen2.5-VL
- Updated for `transformers` ≥ 4.49.0  
- Full Qwen3-VL compatibility (`Qwen/Qwen3-VL-*` series)
- Automatic model downloading through `huggingface_hub`
- Better handling for image batches and temporary file management
- Optional advanced chat node with system/user separation

---

## Installation

Clone or copy this repository into your `ComfyUI/custom_nodes` directory.

Install the required dependencies:

```bash
pip install -r requirements.txt
```