import base64
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ImageDraw
from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from .web_episode import Trajectory

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=True,
)

_CLICK_RADIUS = 12
_CLICK_COLOR = (255, 0, 0)
_CLICK_OUTLINE = (255, 255, 255)
_CLICK_OUTLINE_WIDTH = 3


def _img_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def get_click_xy(step) -> tuple[float, float] | None:
    """Return (x, y) viewport pixel coords if the step's action targets a point."""
    if not step.prediction:
        return None
    action = step.prediction.action
    x = getattr(action, "x", None)
    y = getattr(action, "y", None)
    if x is None or y is None:
        return None
    return (float(x), float(y))


def draw_click_indicator(img, x: float, y: float):
    """Return a copy of *img* with a red click dot burned in at (x, y)."""
    img = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    r = _CLICK_RADIUS
    bbox = [x - r, y - r, x + r, y + r]
    draw.ellipse(bbox, fill=(*_CLICK_COLOR, 140), outline=_CLICK_OUTLINE, width=_CLICK_OUTLINE_WIDTH)
    inner = [x - r // 3, y - r // 3, x + r // 3, y + r // 3]
    draw.ellipse(inner, fill=(*_CLICK_COLOR, 220))
    return img


def annotate_step_image(step):
    """Return the step's screenshot with a click indicator if applicable, or the raw image."""
    if not step.state or not step.state.img:
        return None
    click_xy = get_click_xy(step)
    if click_xy:
        return draw_click_indicator(step.state.img, *click_xy)
    return step.state.img


def _step_context(step) -> dict:
    img = annotate_step_image(step)

    return {
        "img_data_url": _img_to_data_url(img) if img else None,
        "url": step.state.page_url if step.state else "N/A",
        "title": step.state.page_title if step.state else "N/A",
        "prediction_json": json.dumps(step.prediction.model_dump(), indent=2) if step.prediction else None,
        "error": step.error,
    }


def generate_trajectory_html(trajectory: "Trajectory", query: str | None = None) -> str:
    template = _env.get_template("trajectory.html")
    return template.render(
        query=query,
        steps=[_step_context(s) for s in trajectory.steps],
    )


def save_trajectory_html(trajectory: "Trajectory", output_path: str | Path | None = None, query: str | None = None) -> Path:
    if output_path is None:
        default_dir = Path("inference/htmls")
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_dir / "trajectory.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(generate_trajectory_html(trajectory, query), encoding="utf-8")
    return output_path
