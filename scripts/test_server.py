#!/usr/bin/env python3
"""Smoke test for the MolmoWeb model server.

Sends an included screenshot of the Ai2 careers page to a running /predict
endpoint and prints the model's response.

Usage:
    python scripts/test_server.py                          # default: localhost:8001
    python scripts/test_server.py http://myhost:8002       # custom endpoint
"""
import base64
import sys
from pathlib import Path

import requests

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
IMAGE_PATH = Path(__file__).resolve().parent.parent / "assets" / "test_screenshot.png"

PROMPT = (
    "Read the text on this page. "
    "What are the first four job titles listed under 'Open roles'?"
)


def main():
    if not IMAGE_PATH.exists():
        print(f"Error: {IMAGE_PATH} not found.")
        sys.exit(1)

    image_b64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")

    url = f"{ENDPOINT.rstrip('/')}/predict"
    print(f"Endpoint: {url}")
    print(f"Image:    {IMAGE_PATH}")
    print(f"Prompt:   {PROMPT}")
    print()

    try:
        resp = requests.post(
            url,
            json={"prompt": PROMPT, "image_base64": image_b64},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        print("Error: Could not connect to the model server.")
        print(f"Make sure the server is running at {ENDPOINT}")
        sys.exit(1)

    print("Model response:")
    print(resp.json())


if __name__ == "__main__":
    main()
