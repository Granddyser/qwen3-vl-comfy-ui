import os
import uuid
import folder_paths
import numpy as np

from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor,
)
from pathlib import Path
from comfy_api.input import VideoInput

model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)


class DownloadAndLoadQwen3_VLModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen/Qwen3-VL-4B-Instruct",
                        "Qwen/Qwen3-VL-8B-Instruct",
                        "Qwen/Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Thinking",                        
                    ],
                    {"default": "Qwen/Qwen3-VL-4B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "8bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
            },
        }

    RETURN_TYPES = ("QWEN3_VL_MODEL",)
    RETURN_NAMES = ("Qwen3_VL_model",)
    FUNCTION = "DownloadAndLoadQwen3_VLModel"
    CATEGORY = "Qwen3-VL"

    def DownloadAndLoadQwen3_VLModel(self, model, quantization, attention):
        Qwen3_VL_model = {"model": "", "model_path": ""}
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Qwen3VL model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model, local_dir=model_path, local_dir_use_symlinks=False
            )

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        Qwen3_VL_model["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
        )
        Qwen3_VL_model["model_path"] = model_path

        return (Qwen3_VL_model,)


class DownloadAndLoadQwen3_VL_NSFW_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["prithivMLmods/Qwen3-VL-4B-Thinking-abliterated"],
                    {"default": "prithivMLmods/Qwen3-VL-4B-Thinking-abliterated"},
                ),
                "quantization": (["none", "4bit", "8bit"], {"default": "8bit"}),
                "attention": (["flash_attention_2", "sdpa", "eager"], {"default": "sdpa"}),
            },
        }

    RETURN_TYPES = ("QWEN3_VL_NSFW_MODEL",)
    RETURN_NAMES = ("Qwen3_VL_model",)  # gleicher Typ/Name wie normaler Loader
    FUNCTION = "DownloadAndLoadQwen3_VL_NSFW_Model"
    CATEGORY = "Qwen3-VL_NSFW"

    def DownloadAndLoadQwen3_VL_NSFW_Model(self, model, quantization, attention):
        Qwen3_VL_model = {"model": "", "model_path": ""}
        model_name = model.rsplit("/", 1)[-1]
        model_path = os.path.join(model_directory, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading Qwen3VL model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model, local_dir=model_path, local_dir_use_symlinks=False)

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        Qwen3_VL_model["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
        )
        Qwen3_VL_model["model_path"] = model_path
        return (Qwen3_VL_model,)




class Qwen3_VL_Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen3_VL_model": ("QWEN3_VL_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen3_VL_Run"
    CATEGORY = "Qwen3-VL"

    def Qwen3_VL_Run(
        self,
        text,
        Qwen3_VL_model,
        video_decode_method,
        max_new_tokens,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
        BatchImage=None,
    ):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28

        processor = AutoProcessor.from_pretrained(Qwen3_VL_model["model_path"])

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen3_VL_model["model"].device)
        generated_ids = Qwen3_VL_model["model"].generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return (str(output_text[0]),)


class Qwen3_VL_NSFW_Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen3_VL_model": ("QWEN3_VL_NSFW_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen3_VL_NSFW_Run"
    CATEGORY = "Qwen3-VL_NSFW"

    def Qwen3_VL_NSFW_Run(
        self,
        text,
        Qwen3_VL_model,
        video_decode_method,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
        BatchImage=None,
    ):
        import re
        from qwen_vl_utils import process_vision_info

        SYSTEM_PROMPT = (
            """ Generate an explicit and vivid caption for this image that accurately describes all visible elements, including sexual or violent actions. Use colloquial, intense language typical of adult themes, incorporating terms like 'fucking,' 'gets fucked,' 'dick,' 'pussy,' 'cum,' or other appropriate slang to enhance the description. 
                Ensure the tone aligns with the image's content—whether sensual, aggressive, or fetishistic—and highlight specific details such as body positions, clothing, facial expressions, and any explicit acts. Maintain clarity and avoid vague terms."""
        )

        MAX_NEW_TOKENS = 2048  # Fester Tokenwert

        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28

        processor = AutoProcessor.from_pretrained(Qwen3_VL_model["model_path"])

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {"type": "image", "image": uri, "min_pixels": min_pixels, "max_pixels": max_pixels}
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {"type": "image", "image": path, "min_pixels": min_pixels, "max_pixels": max_pixels}
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {"type": "image", "image": path, "min_pixels": min_pixels, "max_pixels": max_pixels}
                )

        if text:
            content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen3_VL_model["model"].device)

        generated_ids = Qwen3_VL_model["model"].generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        text_out = str(output_text[0])
        if "</think>" in text_out:
            text_out = text_out.split("</think>")[-1]

        text_out = re.sub(r"^[\s\u200b\xa0]+", "", text_out)

        return (text_out,)

        
        


class Qwen3_VL_Run_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "BatchImage": ("BatchImage",),
            },
            "required": {
                "system_text": ("STRING", {"default": "", "multiline": True}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "Qwen3_VL_model": ("QWEN3_VL_MODEL",),
                "video_decode_method": (
                    ["torchvision", "decord", "torchcodec"],
                    {"default": "torchvision"},
                ),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1280,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 64,
                        "max": 2048,
                        "tooltip": "Define min_pixels and max_pixels: Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels.",
                    },
                ),
                "total_pixels": (
                    "INT",
                    {
                        "default": 20480,
                        "min": 1,
                        "max": 24576,
                        "tooltip": "We recommend setting appropriate values for the min_pixels and max_pixels parameters based on available GPU memory and the specific application scenario to restrict the resolution of individual frames in the video. Alternatively, you can use the total_pixels parameter to limit the total number of tokens in the video (it is recommended to set this value below 24576 * 28 * 28 to avoid excessively long input sequences). For more details on parameter usage and processing logic, please refer to the fetch_video function in qwen_vl_utils/vision_process.py.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "Qwen3_VL_Run_Advanced"
    CATEGORY = "Qwen3-VL"

    def Qwen3_VL_Run_Advanced(
        self,
        system_text,
        text,
        Qwen3_VL_model,
        video_decode_method,
        max_new_tokens,
        min_pixels,
        max_pixels,
        total_pixels,
        seed,
        image=None,
        video=None,
        BatchImage=None,
    ):
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        total_pixels = total_pixels * 28 * 28

        processor = AutoProcessor.from_pretrained(Qwen3_VL_model["model_path"])

        content = []
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append(
                    {
                        "type": "image",
                        "image": uri,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )
            elif num_counts > 1:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append(
                        {
                            "type": "image",
                            "image": path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        }
                    )

        if video is not None:
            uri = temp_video(video, seed)
            content.append(
                {
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                }
            )

        if BatchImage is not None:
            image_paths = BatchImage
            for path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
                )

        if text:
            content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_decode_method
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = processor(
            text=[modeltext],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(Qwen3_VL_model["model"].device)
        generated_ids = Qwen3_VL_model["model"].generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return (str(output_text[0]),)


class BatchImageLoaderToLocalFiles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("BatchImage",)
    RETURN_NAMES = ("BatchImage",)
    FUNCTION = "BatchImageLoaderToLocalFiles"
    CATEGORY = "Qwen3-VL"

    def BatchImageLoaderToLocalFiles(self, **kwargs):
        images = list(kwargs.values())
        image_paths = []

        for idx, image in enumerate(images):
            unique_id = uuid.uuid4().hex
            image_path = (
                Path(folder_paths.temp_directory) / f"temp_image_{idx}_{unique_id}.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(
                np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            )
            img.save(os.path.join(image_path))

            image_paths.append(f"file://{image_path.resolve().as_posix()}")

        return (image_paths,)


def temp_video(video: VideoInput, seed):
    unique_id = uuid.uuid4().hex
    video_path = (
        Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(
        os.path.join(video_path),
        format="mp4",
        codec="h264",
    )

    uri = f"{video_path.as_posix()}"

    return uri


def temp_image(image, seed):
    unique_id = uuid.uuid4().hex
    image_path = (
        Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    img.save(os.path.join(image_path))

    uri = f"file://{image_path.as_posix()}"

    return uri


def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for Nth_count in range(num_counts):
        img = Image.fromarray(
            np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
        unique_id = uuid.uuid4().hex
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}_{unique_id}.png"
        img.save(os.path.join(image_path))

        image_paths.append(f"file://{image_path.resolve().as_posix()}")

    return image_paths


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQwen3_VLModel": DownloadAndLoadQwen3_VLModel,
    "Qwen3_VL_Run": Qwen3_VL_Run,
    "Qwen3_VL_Run_Advanced": Qwen3_VL_Run_Advanced,
    "BatchImageLoaderToLocalFiles": BatchImageLoaderToLocalFiles,
    "DownloadAndLoadQwen3_VL_NSFW_Model": DownloadAndLoadQwen3_VL_NSFW_Model,
    "Qwen3_VL_NSFW_Run": Qwen3_VL_NSFW_Run, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQwen3 VLModel": "DownloadAndLoadQwen3_VLModel",
    "Qwen3_VL_Run": "Qwen3_VL_Run",
    "Qwen3_VL_Run_Advanced": "Qwen3_VL_Run_Advanced",
    "BatchImageLoaderToLocalFiles": "BatchImageLoaderToLocalFiles",
    "DownloadAndLoadQwen3 VL NSFW Model": "DownloadAndLoadQwen3_VL_NSFW_Model",
    "Qwen3_VL_NSFW_Run": "Qwen3-VL NSFW Run",
}
