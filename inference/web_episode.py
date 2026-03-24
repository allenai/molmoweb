"""
Data models for web episode trajectories (State, Step, Trajectory).
"""
import base64
import io
import os
from pathlib import Path

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from agent.actions import ActionOutput


def save_trajectory_screenshots_png(
    traj: "Trajectory",
    output_dir: str | os.PathLike[str],
    prefix: str = "step",
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, step in enumerate(traj.steps):
        if step.state is None:
            continue
        path = out_dir / f"{prefix}_{str(i).zfill(3)}.png"
        step.state.img.save(path, format="PNG")
        saved.append(str(path))
    return saved


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    img: Image.Image
    page_url: str
    page_title: str

    @field_serializer("img")
    def _serialize_img(self, img: Image.Image, _info):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @field_validator("img", mode="before")
    @classmethod
    def _parse_img(cls, v):
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, (bytes, bytearray)):
            data = bytes(v)
        elif isinstance(v, str):
            prefix = "base64,"
            if v.startswith("data:"):
                idx = v.find(prefix)
                if idx != -1:
                    v = v[idx + len(prefix):]
            data = base64.b64decode(v)
        else:
            raise TypeError(f"img must be PIL.Image or base64 string, got {type(v)!r}")
        img = Image.open(io.BytesIO(data))
        img.load()
        return img


class Step(BaseModel):
    state: State | None
    prediction: ActionOutput | None
    error: str | None


class Trajectory(BaseModel):
    steps: list[Step] = Field(default_factory=list)

    def save_html(self, output_path: str | None = None, query: str | None = None) -> str:
        from .trajectory_visualizer import save_trajectory_html
        return str(save_trajectory_html(self, output_path, query))


