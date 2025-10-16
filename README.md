# Qwen3-VL for ComfyUI

> Implementation for integrating **Qwen3-VL** models into **ComfyUI**, supporting image, multi-image, and video input.

This project is **inspired by and based on prior work** from [makki-shizu](https://github.com/MakkiShizu/), who originally implemented the **Qwen2.5-VL** nodes.  
All code has been updated, rewritten, and adapted for **Qwen3-VL** models.

---

## Description

This custom ComfyUI node set adds native support for **Qwen3-VL** models — the latest multimodal models from Alibaba Cloud’s Qwen team.

Supported input types:
- Text
- Image
- Multi-image
- Video

### Key Differences from Qwen2.5-VL
- Full Qwen3-VL compatibility (`Qwen/Qwen3-VL-*` series)
- Better handling for image batches and temporary file management
- Optional advanced chat node with system/user separation

---

## Installation

Clone or copy this repository into your `ComfyUI/custom_nodes` directory.

Install the required dependencies:

```bash
pip install -r requirements.txt
```