"""
Inference backends for the multimodal agent.

All backends implement predict(prompt, image_np, ...) -> str | None.
Heavy dependencies (torch, olmo) are imported lazily inside each class.
"""
import json
import logging
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI (remote HTTP endpoint)
# ---------------------------------------------------------------------------

class FastApiActionPredictor:
    def __init__(
        self,
        endpoint: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> None:
        self.endpoint = endpoint
        self.temperature = temperature
        self.top_p = top_p

    def predict(self, prompt: str, image_np: np.ndarray, past_actions: list | None = None, **kwargs) -> str | None:
        from utils.vis_utils.image import image_to_base64
        try:
            payload = {"prompt": prompt, "image_base64": image_to_base64(image_np)}
            if past_actions is not None:
                payload["past_actions"] = past_actions
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p

            resp = requests.post(f"{self.endpoint}/predict", json=payload)
            if resp.status_code != 200:
                print(f"[ERROR] FastAPI {self.endpoint} returned {resp.status_code}: {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            print(f"[ERROR] FastAPI {self.endpoint} failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Modal (serverless HTTP endpoint)
# ---------------------------------------------------------------------------

class ModalActionPredictor:
    def __init__(self, endpoint: str, api_key: str | None = None):
        self.endpoint = endpoint
        self.api_key = api_key

    def predict(self, prompt: str, image_np: np.ndarray, **kwargs) -> str | None:
        from utils.vis_utils.image import pil_image_to_base64
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.post(
                self.endpoint,
                headers=headers,
                data=json.dumps({
                    "input_text": [prompt],
                    "input_image": [pil_image_to_base64(Image.fromarray(image_np.astype("uint8")).convert("RGB"))],
                }),
                stream=True,
            )
            if resp.status_code != 200:
                msg = f"Predictor error: Modal {self.endpoint} returned {resp.status_code}: {resp.text}"
                print(msg)
                return msg
            text = ""
            for chunk in resp.iter_lines():
                if chunk:
                    text += json.loads(chunk)["result"]["output"]["text"]
            return text
        except Exception as e:
            msg = f"Predictor error: Modal {self.endpoint} failed: {e}"
            print(msg)
            return msg


# ---------------------------------------------------------------------------
# HuggingFace Transformers (local GPU)
# ---------------------------------------------------------------------------

class HFActionPredictor:
    def __init__(self, checkpoint: str, device: str | None = None, max_new_tokens: int = 1024):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=self.dtype, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, prompt: str, image_np: np.ndarray, **kwargs) -> str | None:
        import torch
        from molmo_utils import process_vision_info

        image = Image.fromarray(image_np.astype("uint8")).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ]}]
        pil_images, _, _ = process_vision_info(messages)
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=pil_images, text=text_input, padding=True, return_tensors="pt").to(self.device)
        inputs = {k: v.to(self.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_tokens = outputs[:, inputs["input_ids"].size(1):]
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        return text or None


# ---------------------------------------------------------------------------
# vLLM (local GPU, high-throughput inference)
# ---------------------------------------------------------------------------

class VLLMActionPredictor:
    """vLLM-backed predictor for fast multimodal inference.

    Requires ``pip install vllm>=0.15.0``.  MolmoWeb models use the same
    Molmo2ForConditionalGeneration architecture that vLLM already supports.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ):
        from vllm import LLM

        # vLLM manages GPU placement internally; `device` is accepted for
        # interface compatibility but not used directly.
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.llm = LLM(
            model=checkpoint,
            trust_remote_code=True,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.9,
        )
        log.info(f"Loaded vLLM model from {checkpoint}")

    def predict(
        self,
        prompt: str,
        image_np: np.ndarray | list[np.ndarray],
        past_actions: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> str | None:
        from vllm import SamplingParams

        # Convert numpy screenshot(s) to PIL
        if isinstance(image_np, list):
            images = [
                Image.fromarray(img.astype("uint8")).convert("RGB")
                for img in image_np
            ]
        else:
            images = [Image.fromarray(image_np.astype("uint8")).convert("RGB")]

        # Build chat message with image(s) + text
        content: list[dict[str, Any]] = []
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

        try:
            outputs = self.llm.chat(messages, sampling_params=sampling_params)
            text = outputs[0].outputs[0].text.strip()
            return text or None
        except Exception as e:
            log.error(f"vLLM inference failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Native OLMo (local GPU, direct model loading)
# ---------------------------------------------------------------------------

class NativeActionPredictor:
    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ):
        import os

        os.environ.setdefault(
            "MOLMO_DATA_DIR",
            os.path.join(os.environ.get("TMPDIR", "/tmp"), "molmo_data"),
        )

        import torch
        from olmo.models.model_config import BaseModelConfig
        from olmo.train.checkpointer import load_model_state
        from olmo.util import resource_path

        self.device = device or "cuda:0"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        cfg_path = resource_path(checkpoint, "config.yaml")
        model_cfg = BaseModelConfig.load(cfg_path, key="model", validate_paths=False)

        with torch.device("meta"):
            self.model = model_cfg.build_model()
        self.model.to_empty(device=self.device)
        load_model_state(checkpoint, self.model)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.preprocessor = model_cfg.build_preprocessor(for_inference=True, is_training=False)
        log.info(f"Loaded native model from {checkpoint} on {self.device}")

    def predict(
        self,
        prompt: str,
        image_np: np.ndarray | list[np.ndarray],
        style: str = "demo",
        past_actions: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        import torch
        from olmo.nn.beam_search import TopPSampler

        image = ([Image.fromarray(img.astype("uint8")).convert("RGB") for img in image_np]
                 if isinstance(image_np, list)
                 else Image.fromarray(image_np.astype("uint8")).convert("RGB"))

        batch = self.preprocessor(dict(image=image, style=style, question=prompt))
        batch["input_ids"] = batch.pop("input_tokens")
        batch.pop("metadata")
        batch = {k: torch.as_tensor(np.expand_dims(v, 0), device=self.device) for k, v in batch.items()}

        sampler = TopPSampler(p=self.top_p, temperature=self.temperature, with_replacement=False)

        if not hasattr(self, "_logged_sampler"):
            print(f"[SAMPLING] TopPSampler(p={self.top_p}, temperature={self.temperature})")
            self._logged_sampler = True

        with torch.inference_mode():
            output = self.model.generate(batch, max_steps=self.max_new_tokens, sampler=sampler, beam_size=1)

        tokens = output.token_ids[0][0]
        result = self.preprocessor.preprocessor.text_preprocessor.tokenizer.decode(tokens).strip()

        return result
