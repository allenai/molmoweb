"""MolmoWeb agent for Qwen/Modal endpoint with inline HTTP calls."""

import base64
import io
import json
from datetime import datetime
from typing import Any, Literal

import numpy as np
import requests
from agent.actions import (
    ALL_ACTIONS,
    ActionOutput,
    BrowserNav,
    GeminiTypeTextAt,
    Goto,
    KeyboardPress,
    KeyboardType,
    MouseClick,
    MouseDragAndDrop,
    Noop,
    ReportInfeasible,
    Scroll,
    ScrollAt,
    SendMsgToUser,
    describe_action,
)
from agent.utils import AgentBase
from jinja2 import Template
from PIL import Image

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
]

MOLMOWEB_THINK_TEMPLATE = Template(
    """
# GOAL
{{ task_description }}

# PREVIOUS STEPS
{% for action in past_actions -%}
## Step {{ action['index'] }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action'] }}
{% endfor %}
# CURRENTLY ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

# NEXT STEP

"""
)


def pil_image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def numpy_to_base64(image_np: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    img = Image.fromarray(image_np.astype("uint8")).convert("RGB")
    return pil_image_to_base64(img)


def convert_action_json_to_action_obj(
    json_action: dict[str, Any] | None
) -> ALL_ACTIONS:
    """Convert action JSON dict to action object."""
    if json_action is None:
        return ReportInfeasible(infeasibility_reason="No action provided")

    action_type = json_action.get("name")

    if action_type in ["click", "dblclick", "mouse_click"]:
        return MouseClick(
            x=float(json_action.get("x", 0.0)),
            y=float(json_action.get("y", 0.0)),
            button=json_action.get("button", "left"),
            click_type="double" if action_type == "dblclick" else "single",
        )
    elif action_type in ["mouse_drag_and_drop"]:
        return MouseDragAndDrop(
            from_x=float(json_action.get("from_x", 0.0)),
            from_y=float(json_action.get("from_y", 0.0)),
            to_x=float(json_action.get("to_x", 0.0)),
            to_y=float(json_action.get("to_y", 0.0)),
        )
    elif action_type == "scroll":
        return Scroll(
            delta_x=float(json_action.get("delta_x", 0.0)),
            delta_y=float(json_action.get("delta_y", 0.0)),
        )
    elif action_type == "scroll_at":
        return ScrollAt(
            x=float(json_action.get("x", 0.0)),
            y=float(json_action.get("y", 0.0)),
            dx=float(json_action.get("delta_x", 0.0)),
            dy=float(json_action.get("delta_y", 0.0)),
        )
    elif action_type in ["type", "keyboard_type"]:
        return KeyboardType(text=json_action.get("text", ""))
    elif action_type in ["keypress", "keyboard_press"]:
        key = json_action.get("key", None)
        lower2key = {k.lower(): k for k in ALLOWED_KEYS}
        if key is None or key.lower() not in lower2key:
            return ReportInfeasible(
                infeasibility_reason=f"Unsupported keypress: {json_action}"
            )
        return KeyboardPress(key=lower2key[key.lower()])
    elif action_type == "gemini_type_text_at":
        return GeminiTypeTextAt(
            x=float(json_action.get("x", 0.0)),
            y=float(json_action.get("y", 0.0)),
            text=json_action.get("text", ""),
            press_enter=json_action.get("press_enter", True),
            clear_before_typing=json_action.get("clear_before_typing", True),
        )
    elif action_type == "goto":
        return Goto(url=json_action.get("url", ""))
    elif action_type == "send_msg_to_user":
        return SendMsgToUser(msg=json_action.get("msg", ""))
    elif action_type == "browser_nav":
        return BrowserNav(
            nav_type=json_action.get("nav_type", "go_back"),
            index=json_action.get("index", -1),
        )
    elif action_type == "noop":
        return Noop(noop_reason=json_action.get("noop_reason", "loading"))
    elif action_type == "report_infeasible":
        return ReportInfeasible(
            infeasibility_reason=json_action.get(
                "infeasibility_reason", "Unknown"
            )
        )
    else:
        return ReportInfeasible(
            infeasibility_reason=f"Unsupported action: {action_type}"
        )


def scale_coordinates(
    x_pct: float, y_pct: float, screenshot: np.ndarray
) -> tuple[float, float]:
    """Scale percentage coordinates (0-100) to pixel coordinates."""
    height, width = screenshot.shape[:2]
    x_pixel = round((x_pct / 100) * width, 1)
    y_pixel = round((y_pct / 100) * height, 1)
    return float(x_pixel), float(y_pixel)


def truncate_str(
    some_str: str, max_len: int, postfix: str = "... (truncated)"
) -> str:
    """Truncate string to max length."""
    if len(some_str) <= max_len:
        return some_str
    return some_str[: max_len - len(postfix)] + postfix


def truncate_urls_or_titles(
    urls_or_titles: list[str] | str | tuple, max_len: int = 100
):
    """Truncate URLs or titles."""
    if isinstance(urls_or_titles, str):
        return truncate_str(urls_or_titles, max_len)
    elif isinstance(urls_or_titles, (list, tuple)):
        return [truncate_str(str(item), max_len) for item in urls_or_titles]
    return truncate_str(str(urls_or_titles), max_len)


class MultimodalAgent(AgentBase):
    """
    MolmoWeb agent for Qwen/Modal endpoint (AGENT_TYPE=molmoweb).

    Self-contained with inline HTTP calls - no external dependencies.
    """

    def __init__(
        self,
        endpoint: str,
        system_message: str = "molmo_web_think",
        api_key: str | None = None,
        inference_mode: Literal["modal", "fastapi"] = "modal",
    ):
        self.endpoint = endpoint
        self.system_message = system_message
        self.inference_mode = inference_mode
        self.api_key = api_key

        self.past_actions: list[dict[str, Any]] = []
        self.past_urls: list[str] = []
        self.last_obs: dict[str, Any] = {}
        self.last_model_inputs: dict[str, Any] | None = None
        self.context: dict[str, Any] | None = None

        self._http_session = requests.Session()

    def reset(self):
        """Reset agent state."""
        self.past_actions = []
        self.past_urls = []
        self.last_obs = {}
        self.last_model_inputs = None
        self.context = None

    def set_context(self, context: dict[str, Any] | None):
        """
        Set context from previous task for follow-up queries.

        Context includes:
        - last_screenshot_base64: Screenshot when previous task completed
        - last_url: URL when previous task completed
        - last_thought: Agent's last thought
        - conversation_history: List of previous queries/actions
        """
        self.context = context

    def _call_modal_endpoint(
        self, prompt: str, screenshot_base64: str
    ) -> str | None:
        """
        Call the Modal model endpoint.

        Returns the raw response text, or None on error.
        """
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "prompt": prompt,
                "image_base64": screenshot_base64,
            }

            print(f"[MolmoWeb] Calling endpoint: {self.endpoint}")

            response = self._http_session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=(30, 300),
            )

            if response.status_code != 200:
                print(
                    f"[MolmoWeb] Error: {response.status_code} - {response.text}"
                )
                return None

            data = response.json()

            # Endpoint may return a plain JSON string (double-encoded)
            # or a dict with result.output.text (streaming format)
            if isinstance(data, str):
                return data
            elif isinstance(data, dict):
                try:
                    return data["result"]["output"]["text"]
                except KeyError:
                    return json.dumps(data)
            else:
                return response.text

        except requests.Timeout:
            print("[MolmoWeb] Request timed out")
            return None
        except Exception as e:
            print(f"[MolmoWeb] Error: {e}")
            return None

    def _call_fastapi_endpoint(
        self, prompt: str, screenshot_base64: str
    ) -> str | None:
        try:
            response = self._http_session.post(
                f"{self.endpoint}/predict",
                json={
                    "prompt": prompt,
                    "image_base64": screenshot_base64,
                },
            )
            if response.status_code != 200:
                print(
                    f"[ERROR] Request to FastAPI endpoint {self.endpoint} failed with status code {response.status_code}:",
                    response.text,
                )
                return None
            else:
                data = response.json()
                print(data)
                # Return a JSON string so predict_action can json.loads consistently with Modal path
                if isinstance(data, str):
                    return data
                return json.dumps(data)

        except Exception as e:
            print(
                f"[ERROR] Request to FastAPI endpoint {self.endpoint} failed: {str(e)}"
            )
            return None

    def _build_prompt(self, obs: dict[str, Any]) -> str:
        """Build the prompt from observation (molmo_web_think template only)."""
        page_index = int(obs["active_page_index"][0])

        user_message = MOLMOWEB_THINK_TEMPLATE.render(
            page_title=truncate_urls_or_titles(
                obs["open_pages_titles"][page_index]
            ),
            page_url=truncate_urls_or_titles(
                obs["open_pages_urls"][page_index]
            ),
            page_index=page_index,
            task_description=obs["goal"],
            past_actions=self.past_actions[-10:],
        )
        return f"{self.system_message}: {user_message}"

    def predict_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Predict the next action given observation.

        Returns:
            (raw_response_text, action_dict)
        """
        prompt = self._build_prompt(obs)
        screenshot_base64 = obs.get("screenshot_base64") or numpy_to_base64(obs["screenshot"])

        # print(f"\n{'='*60}")
        # print(f"📤 [{datetime.now().strftime('%H:%M:%S')}] PROMPT SENT TO API:")
        # print(f"{'='*60}")
        # print(prompt)
        # print(f"{'='*60}\n")

        self.last_model_inputs = {
            "prompt": prompt,
            "url": obs["url"],
            "page_index": int(obs["active_page_index"][0]),
        }

        # Call endpoint with current screenshot
        if self.inference_mode == "modal":
            pred_text = self._call_modal_endpoint(prompt, screenshot_base64)
        elif self.inference_mode == "fastapi":
            pred_text = self._call_fastapi_endpoint(prompt, screenshot_base64)

        # print(
        #     f"🤖 [{datetime.now().strftime('%H:%M:%S')}] Raw response: {pred_text[:200] if pred_text else 'None'}..."
        # )

        # Parse response (Modal returns str; FastAPI may return str or dict)
        try:
            assert pred_text is not None
            pred_json = (
                pred_text
                if isinstance(pred_text, dict)
                else json.loads(pred_text)
            )
            action_json = pred_json["action"]
            thought = pred_json["thought"]
            action_desc = pred_json.get("action_description")
        except Exception as e:
            thought = f"Failed to parse response: {e}"
            action_json = {
                "name": "report_infeasible",
                "infeasibility_reason": f"Parse error: {pred_text}",
            }
            action_desc = None

        # Keep unscaled version for history
        unscaled_action_json = action_json.copy()
        unscaled_action_obj = convert_action_json_to_action_obj(
            unscaled_action_json
        )
        unscaled_action_output = ActionOutput(
            thought=thought, action=unscaled_action_obj
        )

        # Scale coordinates for execution
        action_json = self._scale_action_coordinates(
            action_json, obs["screenshot"]
        )

        # Convert to action object
        action_obj = convert_action_json_to_action_obj(action_json)
        action_output = ActionOutput(thought=thought, action=action_obj)

        # Update history
        self.last_obs = obs
        self.past_actions.append(
            {
                "index": len(self.past_actions) + 1,
                "action": unscaled_action_json,
                "thought": thought,  # truncate_str(thought, max_len=400),
                "action_str": unscaled_action_output.to_str(),
                "action_description": (
                    describe_action(
                        unscaled_action_obj,
                        axtree=None,
                        extra_element_properties=None,
                    )
                    if action_desc is None
                    else truncate_str(action_desc, max_len=400)
                ),
            }
        )
        self.past_urls.append(obs["url"])

        # Build result
        result = {
            "action_output": action_output,
            "thought": truncate_str(thought, max_len=400),
            "action_str": action_output.to_str(),
            "action_description": (
                describe_action(
                    action_obj, axtree=None, extra_element_properties=None
                )
                if action_desc is None
                else truncate_str(action_desc, max_len=400)
            ),
        }

        return pred_text or "", result

    def _scale_action_coordinates(
        self, action_json: dict, screenshot: np.ndarray
    ) -> dict:
        """Scale action coordinates from percentage to pixels."""
        action_json = action_json.copy()
        action_name = action_json.get("name", "")

        if action_name == "click":
            if "x" in action_json and "y" in action_json:
                x, y = scale_coordinates(
                    action_json["x"], action_json["y"], screenshot
                )
                action_json["x"] = x
                action_json["y"] = y
        elif action_name in ["mouse_drag_and_drop", "drag_and_drop"]:
            if "from_x" in action_json and "from_y" in action_json:
                fx, fy = scale_coordinates(
                    action_json["from_x"], action_json["from_y"], screenshot
                )
                action_json["from_x"] = fx
                action_json["from_y"] = fy
            if "to_x" in action_json and "to_y" in action_json:
                tx, ty = scale_coordinates(
                    action_json["to_x"], action_json["to_y"], screenshot
                )
                action_json["to_x"] = tx
                action_json["to_y"] = ty
        elif action_name == "gemini_type_text_at":
            # Gemini uses 1000x1000 grid
            if "x" in action_json and "y" in action_json:
                height, width = screenshot.shape[:2]
                action_json["x"] = round((action_json["x"] / 1000) * width, 1)
                action_json["y"] = round((action_json["y"] / 1000) * height, 1)
        elif action_name in ["scroll", "scroll_at"]:
            if "x" in action_json and "y" in action_json:
                x, y = scale_coordinates(
                    action_json["x"], action_json["y"], screenshot
                )
                action_json["x"] = x
                action_json["y"] = y
            if "delta_x" in action_json and "delta_y" in action_json:
                dx, dy = scale_coordinates(
                    action_json["delta_x"], action_json["delta_y"], screenshot
                )
                action_json["delta_x"] = dx
                action_json["delta_y"] = dy
        elif action_name == "send_msg_to_user":
            action_json["msg"] = truncate_str(
                action_json.get("msg", ""), max_len=1000
            )

        return action_json

    def get_last_model_inputs(self) -> dict[str, Any] | None:
        """Get the last model inputs for debugging."""
        return self.last_model_inputs
