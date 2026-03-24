import json
import os
from typing import Any

from genericpath import exists
from PIL import Image


class LocalEpisodeLogger:
    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def log_image(self, img: Image.Image, fname: str):
        image_dir = os.path.join(self.outdir, "images")
        os.makedirs(image_dir, exist_ok=True)
        img.save(os.path.join(image_dir, fname))

    def write_to_file(self, content: str, fname: str):
        with open(os.path.join(self.outdir, fname), "w") as file:
            file.write(content)

    def log_axtree(self, axtree_str: str, step: int):
        os.makedirs(os.path.join(self.outdir, "axtree"), exist_ok=True)
        axtree_name = f"axtree_{str(step).zfill(3)}.txt"
        self.write_to_file(axtree_str, os.path.join("axtree", axtree_name))
        return axtree_name

    def log_user_message(self, user_message_str: str, step: int):
        os.makedirs(os.path.join(self.outdir, "user_message"), exist_ok=True)
        user_message_name = f"user_message_{str(step).zfill(3)}.txt"
        self.write_to_file(
            user_message_str, os.path.join("user_message", user_message_name)
        )
        return user_message_name

    def log_extra_element_properties(
        self, extra_element_properties: dict, step: int
    ):
        os.makedirs(
            os.path.join(self.outdir, "extra_element_properties"), exist_ok=True
        )

        extra_element_properties_name = (
            f"extra_element_properties_{str(step).zfill(3)}.json"
        )
        self.log_json(
            data={
                k: v
                for k, v in extra_element_properties.items()
                if v["visibility"] == 1
            },
            fname=os.path.join(
                "extra_element_properties", extra_element_properties_name
            ),
            indent=None,
        )
        return extra_element_properties_name

    def log_system_message(self, msg):
        self.write_to_file(content=msg, fname="system_message.md")

    def log_screenshot(self, img: Image.Image, step: int):
        screenshot_name = f"screenshot_{str(step).zfill(3)}.png"
        self.log_image(img, screenshot_name)
        return screenshot_name

    def log_json(self, data: Any, fname: str, indent: int | None = 2):
        with open(os.path.join(self.outdir, fname), "w") as file:
            json.dump(data, file, indent=indent)
