"""
FastAPI server for Qwen 3.5 inference.

Usage:
    CKPT=/weka/oe-training-default/new_peters/qwen/Qwen3.5-9B \\
    uvicorn agent.qwen35_server:app --host 0.0.0.0 --port 8002

Exposes /predict with the same schema as gemma4_server.py:
    POST /predict  {"system_message": "...", "user_message": "...", "image_base64": "..."}
"""

import os
import queue

import torch
from fastapi import FastAPI
from pydantic import BaseModel

CKPT = os.environ.get("CKPT", "/weka/oe-training-default/new_peters/qwen/Qwen3.5-9B")
NUM_PREDICTORS = int(os.environ.get("NUM_PREDICTORS", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
THINKING_MODE = os.environ.get("THINKING_MODE", "0") == "1"


def _build_predictor(ckpt, device, temperature, top_p, max_new_tokens, thinking_mode):
    from agent.qwen35 import Qwen35Predictor
    return Qwen35Predictor(
        checkpoint=ckpt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        torch_dtype="bfloat16",
        thinking_mode=thinking_mode,
    )


def _create_pool(ckpt, n, temperature, top_p, max_new_tokens, thinking_mode) -> queue.Queue:
    pool: queue.Queue = queue.Queue(maxsize=n)
    print(f"[qwen35_server] checkpoint: {ckpt}")
    print(f"[qwen35_server] GPUs: {torch.cuda.device_count()}, predictors: {n}")
    for i in range(n):
        device = f"cuda:{i}"
        pred = _build_predictor(ckpt, device, temperature, top_p, max_new_tokens, thinking_mode)
        print(f"[qwen35_server] predictor {i} ready on {device}")
        pool.put(pred)
    return pool


predictor_pool = _create_pool(
    CKPT, NUM_PREDICTORS, TEMPERATURE, TOP_P, MAX_NEW_TOKENS, THINKING_MODE
)

app = FastAPI()


class PredictRequest(BaseModel):
    system_message: str
    user_message: str
    image_base64: str | None = None
    temperature: float | None = None
    top_p: float | None = None


@app.post("/predict")
def predict(request: PredictRequest):
    from PIL import Image
    from utils.vis_utils.image import base64_to_numpy_image

    image = None
    if request.image_base64:
        img_np = base64_to_numpy_image(request.image_base64)
        image = Image.fromarray(img_np.astype("uint8")).convert("RGB")

    try:
        pred = predictor_pool.get(timeout=60)
    except queue.Empty:
        return {"error": "All predictors busy"}

    saved_temp, saved_top_p = pred.temperature, pred.top_p
    try:
        if request.temperature is not None:
            pred.temperature = request.temperature
        if request.top_p is not None:
            pred.top_p = request.top_p
        result = pred.predict(
            system_message=request.system_message,
            user_message=request.user_message,
            image=image,
        )
    except Exception as e:
        result = f"Predictor error: {e}"
    finally:
        pred.temperature = saved_temp
        pred.top_p = saved_top_p
        predictor_pool.put(pred)

    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run("agent.qwen35_server:app", host="0.0.0.0", port=port, reload=False)
