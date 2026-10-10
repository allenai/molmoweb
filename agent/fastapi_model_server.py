import os

# Set before any `olmo` import (lazy in model_backends). Public molmo2 warns if unset.
os.environ.setdefault("MOLMO_DATA_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "molmo_data"))

import queue
import torch
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)

from fastapi import FastAPI
from pydantic import BaseModel

from agent.model_backends import HFActionPredictor, NativeActionPredictor
from utils.vis_utils.image import base64_to_numpy_image


CKPT = os.environ.get("CKPT")
if CKPT is None:
    print("Warning: environment variable CKPT is not set")

NUM_PREDICTORS = int(os.environ.get("NUM_PREDICTORS", "1"))
PREDICTOR_TYPE = os.environ.get("PREDICTOR_TYPE", "native")

TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.8"))


def create_predictor_pool(
    ckpt: str,
    num_predictors: int = 1,
    predictor_type: str = "native",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
) -> queue.Queue:
    pool: queue.Queue = queue.Queue(maxsize=num_predictors)

    print(f"Using checkpoint: {ckpt}")
    
    # Determine device(s)
    FORCE_DEVICE = os.environ.get("DEVICE")
    if FORCE_DEVICE:
        print(f"Forcing device: {FORCE_DEVICE}")
        devices = [FORCE_DEVICE] * num_predictors
    elif torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}, predictors: {num_predictors}, type: {predictor_type}")
        devices = [f"cuda:{i % torch.cuda.device_count()}" for i in range(num_predictors)]
    elif torch.backends.mps.is_available():
        print(f"Apple Silicon (MPS) detected, predictors: {num_predictors}, type: {predictor_type}")
        devices = ["mps"] * num_predictors
    else:
        print(f"No GPU found, using CPU, predictors: {num_predictors}, type: {predictor_type}")
        devices = ["cpu"] * num_predictors

    for i in range(num_predictors):
        device = devices[i]

        if predictor_type == "hf":
            predictor = HFActionPredictor(checkpoint=ckpt, device=device)
        elif predictor_type == "native":
            predictor = NativeActionPredictor(
                checkpoint=ckpt, device=device,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p,
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        print(f"Created {type(predictor).__name__} on {device}")
        pool.put(predictor)

    return pool


predictor_pool = create_predictor_pool(
    ckpt=CKPT,
    num_predictors=NUM_PREDICTORS,
    predictor_type=PREDICTOR_TYPE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
)

app = FastAPI()


class PredictRequest(BaseModel):
    prompt: str
    image_base64: str
    past_actions: list | None = None
    temperature: float | None = None
    top_p: float | None = None


@app.post("/predict")
def predict(request: PredictRequest):
    global predictor_pool
    image_np = base64_to_numpy_image(request.image_base64)

    try:
        predictor = predictor_pool.get(timeout=30)
    except queue.Empty:
        return "Predictor error: All predictors are busy"

    try:
        saved = {
            "temperature": getattr(predictor, "temperature", None),
            "top_p": getattr(predictor, "top_p", None),
        }
        if request.temperature is not None:
            predictor.temperature = request.temperature
        if request.top_p is not None:
            predictor.top_p = request.top_p

        try:
            result = predictor.predict(request.prompt, image_np, past_actions=request.past_actions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for k, v in saved.items():
                if v is not None:
                    setattr(predictor, k, v)
    finally:
        predictor_pool.put(predictor)
    return result
