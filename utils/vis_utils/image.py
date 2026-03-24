import base64
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image


def pil_image_to_base64(pil_image: Image.Image, format="PNG"):
    """
    Encode a PIL Image to base64 string

    Args:
        pil_image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def numpy_image_to_base64(np_image: np.ndarray, format="PNG"):
    img = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")
    return pil_image_to_base64(img, format)


def image_to_base64(image: np.ndarray | Image.Image, format="PNG"):
    if isinstance(image, np.ndarray):
        return numpy_image_to_base64(image, format)
    elif isinstance(image, Image.Image):
        return pil_image_to_base64(image, format)
    else:
        raise TypeError("image must be a numpy array or PIL Image")


def base64_to_pil_image(base64_string: str):
    # Remove the header if present (e.g., "data:image/png;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    # Decode the base64 string
    img_data = base64.b64decode(base64_string)
    # Open the image
    return Image.open(BytesIO(img_data)).convert("RGB")


def base64_to_numpy_image(base64_string: str):
    return np.array(base64_to_pil_image(base64_string))


def base64_to_image(base64_string: str, format=Literal["pil", "numpy"]):
    if format == "pil":
        return base64_to_pil_image(base64_string)
    elif format == "numpy":
        return base64_to_numpy_image(base64_string)
    else:
        raise ValueError("format must be 'pil' or 'numpy'")


def pil_image_to_html(pil_image: Image.Image):
    base64_image = pil_image_to_base64(pil_image)
    html = f'<img src="data:image/png;base64,{base64_image}" alt="Image">'
    return html
