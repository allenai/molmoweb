import json
import os
from datetime import datetime
from typing import Any, Literal

import numpy as np
from jinja2 import Template
from agent.actions import (
    ALL_ACTIONS,
    ActionOutput,
    BrowserNav,
    GeminiTypeTextAt,
    Goto,
    HoverAt,
    KeyboardPress,
    KeyboardType,
    MouseClick,
    MouseDragAndDrop,
    Noop,
    ReportInfeasible,
    Scroll,
    ScrollAt,
    SendMsgToUser,
)
from agent.utils import AgentBase

ALLOWED_KEYS = [
    "Enter",
    "Escape",
    "Backspace",
    "Tab",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "ControlOrMeta+a",
    "ControlOrMeta+c",
    "ControlOrMeta+v",
    "F5",
]

USER_MSG_TEMPLATE = Template(
    """
# GOAL
{{ task_description }}

# PREVIOUS STEPS
{% for action in past_actions: -%}
## Step {{ action['index'] }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action'] }}
{% endfor %}
# CURRENTLY ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

# NEXT STEP

"""
)


def _pct_to_px(pct: float, dim: int) -> float:
    return round((pct / 100.0) * dim, 1)


def _pct_to_coord(pct: float, dim: int) -> float:
    """Convert percentage to pixel coordinate, clamped to [1, dim-2] so edge
    predictions still land inside the viewport."""
    px = round((pct / 100.0) * dim, 1)
    return max(1.0, min(px, dim - 2.0))


def convert_action_json_to_action_obj(
    json_action: dict[str, Any] | None,
    screenshot: np.ndarray | None = None,
) -> ALL_ACTIONS:
    """Convert a JSON action dict to a pydantic action object.

    Coordinates are expected as percentages (0-100) and converted to pixels
    using the screenshot dimensions.
    """
    if json_action is None:
        return ReportInfeasible(infeasibility_reason=f"Unsupported action: {json_action}")

    action_type = json_action.get("name")
    h, w = (screenshot.shape[:2] if screenshot is not None else (720, 1280))

    if action_type in ["click", "dblclick", "mouse_click"]:
        return MouseClick(
            x=_pct_to_coord(float(json_action.get("x", 0.0)), w),
            y=_pct_to_coord(float(json_action.get("y", 0.0)), h),
            button=json_action.get("button", "left"),
            click_type="double" if action_type == "dblclick" else "single",
        )
    elif action_type == "hover_at":
        return HoverAt(
            x=_pct_to_coord(float(json_action.get("x", 0.0)), w),
            y=_pct_to_coord(float(json_action.get("y", 0.0)), h),
            duration=float(json_action.get("duration", 1.0)),
        )
    elif action_type in ["drag_and_drop", "mouse_drag_and_drop"]:
        return MouseDragAndDrop(
            from_x=_pct_to_coord(float(json_action.get("from_x", 0.0)), w),
            from_y=_pct_to_coord(float(json_action.get("from_y", 0.0)), h),
            to_x=_pct_to_coord(float(json_action.get("to_x", 0.0)), w),
            to_y=_pct_to_coord(float(json_action.get("to_y", 0.0)), h),
        )
    elif action_type == "scroll":
        return Scroll(
            delta_x=_pct_to_px(float(json_action.get("delta_x", 0.0)), w),
            delta_y=_pct_to_px(float(json_action.get("delta_y", 0.0)), h),
        )
    elif action_type == "scroll_at":
        return ScrollAt(
            x=_pct_to_coord(float(json_action.get("x", 0.0)), w),
            y=_pct_to_coord(float(json_action.get("y", 0.0)), h),
            delta_x=_pct_to_px(float(json_action.get("delta_x", 0.0)), w),
            delta_y=_pct_to_px(float(json_action.get("delta_y", 0.0)), h),
        )
    elif action_type in ("type", "keyboard_type"):
        return KeyboardType(text=json_action.get("text", ""))
    elif action_type in ("keypress", "keyboard_press"):
        key = json_action.get("key", None)
        lower2key = {k.lower(): k for k in ALLOWED_KEYS}
        if key is None or key.lower() not in lower2key:
            return ReportInfeasible(infeasibility_reason=f"Unsupported keypress: {json_action}")
        return KeyboardPress(key=lower2key[key.lower()])
    elif action_type == "gemini_type_text_at":
        return GeminiTypeTextAt(
            x=_pct_to_coord(float(json_action.get("x", 0.0)), w),
            y=_pct_to_coord(float(json_action.get("y", 0.0)), h),
            text=json_action.get("text", ""),
            press_enter=json_action.get("press_enter", True),
            clear_before_typing=json_action.get("clear_before_typing", True),
        )
    elif action_type == "goto":
        return Goto(url=json_action.get("url", ""))
    elif action_type == "send_msg_to_user":
        return SendMsgToUser(msg=json_action.get("msg", ""))
    elif action_type == "browser_nav":
        return BrowserNav(nav_type=json_action.get("nav_type", "go_back"), index=json_action.get("index", -1))
    elif action_type == "noop":
        return Noop(noop_reason=json_action.get("noop_reason", "loading"))
    elif action_type == "report_infeasible":
        return ReportInfeasible(infeasibility_reason=json_action.get("infeasibility_reason", "Infeasibility reason unavailable"))
    else:
        return ReportInfeasible(infeasibility_reason=f"Unsupported action type: {action_type}. Predicted: {json_action}")


def truncate_str(s: str, max_len: int, postfix: str = "... (truncated)") -> str:
    return s if len(s) <= max_len else s[: max_len - len(postfix)] + postfix


def truncate_urls_or_titles(urls_or_titles: list[str] | str | tuple, max_len: int = 100):
    if isinstance(urls_or_titles, str):
        return truncate_str(urls_or_titles, max_len)
    elif isinstance(urls_or_titles, (list, tuple)):
        return [truncate_str(str(item), max_len) for item in urls_or_titles]
    return truncate_str(str(urls_or_titles), max_len)


class MultimodalAgent(AgentBase):
    def __init__(
        self,
        endpoint_or_checkpoint: str,
        system_message: str = "molmo_web_think",
        inference_mode: Literal["local", "fastapi", "modal", "native"] = "fastapi",
        device: str | None = None,
        api_key: str | None = None,
        max_past_steps: int = 3,
        max_past_images: int = 0,
        sampling_temperature: float = 0.7,
        sampling_top_p: float = 0.8,
    ):
        self.endpoint_or_checkpoint = endpoint_or_checkpoint
        self.system_message = system_message
        self.inference_mode = inference_mode
        self.device = device
        self.api_key = api_key
        self.max_past_steps = max_past_steps
        self.max_past_images = max_past_images
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p

        self.past_actions: list[dict[str, Any]] = []
        self.past_observations: list[dict[str, Any]] = []

        if inference_mode == "fastapi":
            from agent.model_backends import FastApiActionPredictor
            self.predictor = FastApiActionPredictor(
                endpoint=self.endpoint_or_checkpoint,
                temperature=self.sampling_temperature,
                top_p=self.sampling_top_p,
            )
        elif inference_mode == "modal":
            from agent.model_backends import ModalActionPredictor
            self.predictor = ModalActionPredictor(endpoint=self.endpoint_or_checkpoint, api_key=self.api_key)
        elif inference_mode == "native":
            from agent.model_backends import NativeActionPredictor
            self.predictor = NativeActionPredictor(
                checkpoint=self.endpoint_or_checkpoint, device=self.device,
                temperature=self.sampling_temperature,
                top_p=self.sampling_top_p,
            )
        else:
            raise ValueError(f"Invalid inference_mode: {inference_mode}")
        self.last_model_inputs: dict[str, Any] | None = None

    def reset(self):
        self.past_actions = []
        self.past_observations = []
        self.last_model_inputs = None

    def get_user_message(self, obs: dict[str, Any]) -> str:
        page_index = int(obs["active_page_index"][0])

        user_message = USER_MSG_TEMPLATE.render(
            page_title=truncate_urls_or_titles(obs["open_pages_titles"][page_index]),
            page_url=truncate_urls_or_titles(obs["open_pages_urls"][page_index]),
            page_index=page_index,
            task_description=obs["goal"],
            past_actions=self.past_actions[-self.max_past_steps:],
        )

        return user_message

    def predict_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        user_message = self.get_user_message(obs)
        prompt = f"{self.system_message}: {user_message}"

        if self.max_past_images > 0:
            past_images = [
                po["screenshot"] for po in self.past_observations[-self.max_past_images:]
                if "screenshot" in po
            ]
            image = past_images + [obs["screenshot"]]
        else:
            image = obs["screenshot"]

        past_actions_dict = [
            {"thought": a["thought"], "action": a["action"]}
            for a in self.past_actions[-self.max_past_steps:]
        ]
        pred_text: str | None = self.predictor.predict(
            prompt=prompt,
            image_np=image,
            past_actions=past_actions_dict,
        )
        # print(f"[{datetime.now().strftime('%H:%M:%S')}] Raw predicted text: {pred_text}")

        self.last_model_inputs = {
            "system_message": self.system_message,
            "user_message": user_message,
            "prompt": prompt,
            "image_np": obs["screenshot"],
            "page_index": int(obs["active_page_index"][0]),
            "url": obs["url"],
            "open_pages_titles": obs["open_pages_titles"],
        }

        try:
            assert pred_text is not None
            pred_json: dict[str, Any] = json.loads(pred_text)
            if "action" in pred_json:
                action_json: dict[str, Any] = pred_json["action"]
                thought: str = pred_json.get("thought", "")
                action_desc = pred_json.get("action_description", None)
            elif "name" in pred_json:
                action_json = pred_json
                thought = ""
                action_desc = None
            else:
                raise ValueError(f"Expected 'action' or 'name' key in parsed JSON but didnt get it.")
        except Exception as e:
            pred_json = dict(
                thought=f"Could not parse predicted action: {e}",
                action=dict(name="report_infeasible", infeasibility_reason=f"Unparseable model output: {pred_text}"),
            )
            action_json = pred_json["action"]
            thought = pred_json["thought"]
            action_desc = None

        if action_json.get("name") == "gemini_type_text_at":
            if "x" in action_json and "y" in action_json:
                action_json["x"] = action_json["x"] / 10.0
                action_json["y"] = action_json["y"] / 10.0

        if action_json.get("name") == "send_msg_to_user":
            action_json["msg"] = truncate_str(action_json.get("msg", ""), max_len=1000)

        action_obj = convert_action_json_to_action_obj(action_json, screenshot=obs["screenshot"])
        action_output = ActionOutput(thought=thought, action=action_obj)

        desc = action_output.describe() if action_desc is None else truncate_str(action_desc, max_len=400)

        next_action = {
            "action_output": action_output,
            "thought": thought,
            "action_str": action_output.to_str(),
            "action": action_json,
            "action_description": desc,
        }
        self.past_observations.append(obs)
        self.past_actions.append(
            {
                "index": len(self.past_actions) + 1,
                "action_output": action_output,
                "thought": thought,
                "action_str": action_output.to_str(),
                "action": action_json,
                "action_description": desc,
            }
        )
        return pred_text, next_action

    def get_last_model_inputs(self) -> dict[str, Any] | None:
        return self.last_model_inputs
