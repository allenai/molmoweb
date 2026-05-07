"""
Qwen 3.5 web agent.

Runs Qwen3.5-9B (or any Qwen3/Qwen3.5 variant) locally via HuggingFace
Transformers and parses its text output into structured browser actions.

Qwen3.5 is a multimodal instruction model that supports an optional
chain-of-thought reasoning mode where outputs are wrapped in
<think>...</think> tags before the actual response.  The agent works in
both thinking mode (richer reasoning, slower) and non-thinking mode
(faster, /no_think appended to the prompt).

Action parsing strips <think> blocks first, then applies the same
multi-strategy JSON extraction used by the Gemma 4 agent:
  1. direct JSON parse
  2. extract from ```json ... ``` code fences
  3. find first { ... } block
  4. fall back to report_infeasible
"""

import json
import re
from typing import Any, Literal

import numpy as np
from jinja2 import Template
from PIL import Image

from agent.actions import (
    ALL_ACTIONS,
    ActionOutput,
    BrowserNav,
    Click,
    Goto,
    KeyboardPress,
    KeyboardType,
    MouseClick,
    Noop,
    ReportInfeasible,
    Scroll,
    SendMsgToUser,
)
from agent.utils import AgentBase

# ---------------------------------------------------------------------------
# Allowed keys
# ---------------------------------------------------------------------------

ALLOWED_KEYS = [
    "Enter", "Escape", "Backspace", "Tab",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "ControlOrMeta+a", "ControlOrMeta+c", "ControlOrMeta+v", "F5",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """You are an intelligent web assistant. You are given a web page represented as an accessibility tree (unique ids called bid for each DOM element) and optionally a screenshot. Your job is to predict the next browser action to accomplish the user's GOAL.

# Page representation
Elements are shown as: [bid] role 'name'

# Response format
Output ONLY valid JSON — no prose before or after. Schema:
{
  "thought": "<brief high-level reasoning>",
  "action": {"name": "<action_name>", ...fields...}
}

# Available actions

## click — single left-click on an element
{"name": "click", "bid": "<element_id>"}
## dblclick — double-click
{"name": "dblclick", "bid": "<element_id>"}
## type — keyboard input AT the currently focused element (MUST click input first)
{"name": "type", "text": "<text>"}
## keypress
{"name": "keypress", "key": "<key>"}
  Allowed keys: Enter, Escape, Backspace, Tab, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, ControlOrMeta+a, ControlOrMeta+c, ControlOrMeta+v, F5
## scroll — scroll the page (delta_y positive = down)
{"name": "scroll", "delta_x": 0, "delta_y": 720}
## goto — navigate to a URL
{"name": "goto", "url": "<url>"}
## go_back
{"name": "browser_nav", "nav_type": "go_back", "index": -1}
## noop — wait for page to load
{"name": "noop", "noop_reason": "loading"}
## send answer to user
{"name": "send_msg_to_user", "msg": "<message>"}
## report infeasible task
{"name": "report_infeasible", "infeasibility_reason": "<reason>"}

# CRITICAL RULES

## Input fields — always 3 separate steps:
1. {"name": "click", "bid": "<input_bid>"}   ← click to focus
2. {"name": "type", "text": "<your text>"}   ← type (only works after focus)
3. {"name": "keypress", "key": "Enter"}      ← or click submit button

NEVER issue a type action without first clicking the input field in the previous step.

## Navigation
- Use goto only for URLs visible on the page, given in the task, or well-known homepages.
- Never construct or guess URLs with paths or query parameters.
- If a page URL did not change after submitting, the field may not have been focused — click the input and retry.

## Finishing
1. {"name": "send_msg_to_user", "msg": "[ANSWER] <single concise answer>"}
2. {"name": "send_msg_to_user", "msg": "[EXIT]"}

## Thought
- Concise and high-level — do NOT mention bids, DOM structure, HTML, or accessibility trees.
- State: what was done, what the page shows now, why this action is next.
"""

# ---------------------------------------------------------------------------
# User message template
# ---------------------------------------------------------------------------

USER_MSG_TEMPLATE = Template(
    """# GOAL
{{ task_description }}

# PREVIOUS ACTIONS
{% for action in past_actions -%}
## Step {{ loop.index }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action_str'] }}
DESCRIPTION: {{ action['action_description'] }}
{% endfor %}

# CURRENT OBSERVATION

## OPEN TABS
{% for title, url in open_pages_titles_and_urls -%}
    Page {{ loop.index-1 }}: {{ title }} | {{ url }}
{% endfor %}

## ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

## ACCESSIBILITY TREE
{{ axtree_str }}

## LAST ACTION ERROR
{{ last_action_error }}
"""
)

# ---------------------------------------------------------------------------
# Qwen-specific output parsing
# ---------------------------------------------------------------------------

def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from Qwen output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _try_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _from_fences(text: str) -> dict | None:
    for pat in [r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            r = _try_json(m.group(1).strip())
            if r is not None:
                return r
    return None


def _first_brace_block(text: str) -> dict | None:
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                r = _try_json(text[start : i + 1])
                if r is not None:
                    return r
                start = None
    return None


def _normalize(action: dict) -> dict:
    """Normalise Qwen action-name and field aliases."""
    a = action.copy()

    # Name field aliases
    if "name" not in a:
        for k in ("action_type", "type", "action"):
            if k in a and isinstance(a[k], str):
                a["name"] = a.pop(k)
                break

    name = a.get("name", "")
    _aliases = {
        "mouse_click": "click", "left_click": "click", "right_click": "click",
        "single_click": "click", "double_click": "dblclick",
        "keyboard_type": "type", "input_text": "type", "type_text": "type",
        "enter_text": "type", "fill": "type",
        "keyboard_press": "keypress", "press_key": "keypress", "key_press": "keypress",
        "navigate": "goto", "go_to": "goto", "open_url": "goto", "open": "goto",
        "send_message": "send_msg_to_user", "answer": "send_msg_to_user",
        "report": "report_infeasible",
        "wait": "noop", "wait_for_load": "noop",
        "go_back": "browser_nav",
    }
    a["name"] = _aliases.get(name, name)

    # go_back shorthand
    if a["name"] == "go_back" and "nav_type" not in a:
        a["nav_type"] = "go_back"
        a["index"] = -1

    # Field aliases
    for src, dst in [("content", "text"), ("value", "text"), ("query", "text"),
                     ("element_id", "bid"), ("element", "bid"),
                     ("message", "msg"), ("reason", "infeasibility_reason")]:
        if src in a and dst not in a:
            a[dst] = a.pop(src)

    return a


def _to_action_obj(action_json: dict) -> ALL_ACTIONS:
    action_json = _normalize(action_json)
    name = action_json.get("name", "")

    if name in ("click", "dblclick"):
        bid = action_json.get("bid")
        if bid:
            return Click(
                bid=str(bid),
                button=action_json.get("button", "left"),
                click_type="double" if name == "dblclick" else "single",
            )
        x, y = float(action_json.get("x", 0)), float(action_json.get("y", 0))
        return MouseClick(x=x, y=y, button=action_json.get("button", "left"),
                          click_type="double" if name == "dblclick" else "single")

    if name == "type":
        return KeyboardType(text=action_json.get("text", ""))

    if name == "keypress":
        key = action_json.get("key", "")
        lower2key = {k.lower(): k for k in ALLOWED_KEYS}
        if key.lower() in lower2key:
            return KeyboardPress(key=lower2key[key.lower()])
        return ReportInfeasible(infeasibility_reason=f"Unsupported key: {key!r}")

    if name == "scroll":
        return Scroll(delta_x=float(action_json.get("delta_x", 0)),
                      delta_y=float(action_json.get("delta_y", 0)))

    if name == "goto":
        return Goto(url=action_json.get("url", ""))

    if name == "browser_nav":
        return BrowserNav(nav_type=action_json.get("nav_type", "go_back"),
                          index=int(action_json.get("index", -1)))

    if name == "noop":
        reason = action_json.get("noop_reason", "loading")
        valid = {"loading", "captcha", "unsupported_keypress", "retrying_after_api_error"}
        return Noop(noop_reason=reason if reason in valid else "loading")

    if name == "send_msg_to_user":
        return SendMsgToUser(msg=action_json.get("msg", ""))

    if name == "report_infeasible":
        return ReportInfeasible(
            infeasibility_reason=action_json.get("infeasibility_reason", "Infeasible"))

    return ReportInfeasible(infeasibility_reason=f"Unknown action: {name!r}")


def parse_qwen_output(text: str) -> tuple[dict, str, str]:
    """
    Parse Qwen's raw output into (action_json, thought, raw_cleaned).

    Strips <think>…</think> first, then extracts JSON using three strategies:
      1. direct parse of the cleaned text
      2. extract from ```json…``` or ```…``` fences
      3. first valid {…} block in the text
    """
    raw = text.strip() if text else ""
    cleaned = strip_think_blocks(raw)

    parsed = (
        _try_json(cleaned)
        or _from_fences(cleaned)
        or _first_brace_block(cleaned)
        # Fallback: try the raw text in case think-block stripping broke something
        or _try_json(raw)
        or _from_fences(raw)
        or _first_brace_block(raw)
    )

    if parsed is None:
        return (
            {"name": "report_infeasible",
             "infeasibility_reason": f"Unparseable output: {cleaned[:300]}"},
            "",
            cleaned,
        )

    if "action" in parsed and isinstance(parsed["action"], dict):
        thought = parsed.get("thought", "")
        action_json = parsed["action"]
    elif "name" in parsed:
        thought = parsed.get("thought", "")
        action_json = {k: v for k, v in parsed.items() if k != "thought"}
    else:
        thought = parsed.get("thought", "")
        action_json = {"name": "report_infeasible",
                       "infeasibility_reason": f"No action key: {parsed}"}

    return action_json, thought, cleaned


# ---------------------------------------------------------------------------
# Local model predictor
# ---------------------------------------------------------------------------

class Qwen35Predictor:
    """Runs Qwen3.5 locally using HuggingFace Transformers."""

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        torch_dtype: str = "bfloat16",
        thinking_mode: bool = False,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_mode = thinking_mode

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        print(f"[Qwen35] Loading processor from {checkpoint}")
        self.processor = AutoProcessor.from_pretrained(checkpoint)

        print(f"[Qwen35] Loading model on {self.device} dtype={torch_dtype}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()
        print("[Qwen35] Model loaded.")

    def predict(
        self,
        system_message: str,
        user_message: str,
        image: Image.Image | None = None,
    ) -> str:
        import torch

        content: list[dict] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user_message})

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content},
        ]

        # enable_thinking=False → template inserts empty <think></think> block,
        # suppressing chain-of-thought. enable_thinking=True starts a real <think> block.
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking_mode,
        )

        if image is not None:
            inputs = self.processor(
                text=text_input, images=[image], return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=text_input, return_tensors="pt"
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=self.top_p,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Qwen35Agent
# ---------------------------------------------------------------------------

class Qwen35Agent(AgentBase):
    """
    Web agent powered by a local Qwen 3.5 model.

    Uses the accessibility tree for element targeting (bid-based).
    Optionally includes a screenshot for visual grounding.
    Handles Qwen's <think>…</think> chain-of-thought output transparently.
    """

    def __init__(
        self,
        checkpoint: str = "/weka/oe-training-default/new_peters/qwen/Qwen3.5-9B",
        device: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        torch_dtype: str = "bfloat16",
        thinking_mode: bool = False,
        include_screenshot: bool = True,
        system_message: str = SYSTEM_MESSAGE,
        max_past_steps: int = 10,
    ):
        self.include_screenshot = include_screenshot
        self.system_message = system_message
        self.max_past_steps = max_past_steps

        self.predictor = Qwen35Predictor(
            checkpoint=checkpoint,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            torch_dtype=torch_dtype,
            thinking_mode=thinking_mode,
        )

        self.past_actions: list[dict[str, Any]] = []
        self.past_urls: list[str] = []
        self.last_model_inputs: dict[str, Any] | None = None

    def reset(self):
        self.past_actions = []
        self.past_urls = []
        self.last_model_inputs = None

    def _get_axtree_str(self, obs: dict[str, Any]) -> str:
        from utils.axtree import flatten_axtree_to_str
        return flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=obs.get("extra_element_properties"),
            with_visible=False,
            filter_visible_only=True,
            filter_with_bid_only=True,
            with_clickable=True,
            skip_generic=True,
        )

    def _build_user_message(self, obs: dict[str, Any], axtree_str: str) -> str:
        page_index = int(obs["active_page_index"][0])
        err = obs.get("last_action_error", "") or ""
        if "TimeoutError" in err:
            err = ""
        err = err or "The action was successful with no error."

        return USER_MSG_TEMPLATE.render(
            task_description=obs["goal"],
            past_actions=self.past_actions[-self.max_past_steps:],
            open_pages_titles_and_urls=zip(
                obs["open_pages_titles"], obs["open_pages_urls"]
            ),
            page_index=page_index,
            page_title=obs["open_pages_titles"][page_index],
            page_url=obs["open_pages_urls"][page_index],
            axtree_str=axtree_str,
            last_action_error=err,
        )

    def predict_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        axtree_str = self._get_axtree_str(obs)
        user_message = self._build_user_message(obs, axtree_str)

        image: Image.Image | None = None
        if self.include_screenshot and obs.get("screenshot") is not None:
            image = Image.fromarray(obs["screenshot"].astype("uint8")).convert("RGB")

        raw_text = self.predictor.predict(
            system_message=self.system_message,
            user_message=user_message,
            image=image,
        )

        action_json, thought, cleaned = parse_qwen_output(raw_text)
        action_obj = _to_action_obj(action_json)
        action_output = ActionOutput(thought=thought, action=action_obj)

        self.last_model_inputs = {
            "axtree_str": axtree_str,
            "system_message": self.system_message,
            "user_message": user_message,
            "page_index": int(obs["active_page_index"][0]),
            "url": obs["url"],
            "open_pages_titles": obs["open_pages_titles"],
            "open_pages_urls": obs["open_pages_urls"],
        }

        step = {
            "action_output": action_output,
            "thought": thought,
            "action_str": action_output.to_str(),
            "action_description": action_output.describe(
                axtree=obs.get("axtree_object"),
                extra_element_properties=obs.get("extra_element_properties"),
            ),
            "raw_output": cleaned,
        }
        self.past_actions.append(step)
        self.past_urls.append(obs["url"])

        return raw_text, step

    def get_last_model_inputs(self) -> dict[str, Any] | None:
        return self.last_model_inputs
