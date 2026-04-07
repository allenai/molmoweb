import json
from datetime import datetime
from typing import Any

import numpy as np
import fasthtml.common as ft
from PIL import Image, ImageDraw

from agent.actions import get_node_properties
from utils.eval_utils.episode import Interaction
from utils.eval_utils.episode_logger import LocalEpisodeLogger
from utils.vis_utils.html import create_page, save_html
from utils.vis_utils.vis import *


KEYS_TO_STRIP = {"axtree_str", "system_message", "user_message", "image_np"}


def _strip_non_serializable(obj: Any) -> Any:
    """Recursively strip numpy arrays and other non-JSON-serializable values."""
    if isinstance(obj, dict):
        return {
            k: _strip_non_serializable(v)
            for k, v in obj.items()
            if k not in KEYS_TO_STRIP and not isinstance(v, np.ndarray)
        }
    if isinstance(obj, list):
        return [_strip_non_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return None
    return obj


def stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return str(v)


def log_episode(
    interactions: list[Interaction],
    metadata: dict[str, Any],
    system_message: str,
    outdir: str,
    instruction: str,
    task_type: str,
    bb_session_id: str | None = None,
):
    system_message = stringify(system_message)
    instruction = stringify(instruction)
    task_type = stringify(task_type)

    logger = LocalEpisodeLogger(outdir)
    trajectory = {}

    logger.log_system_message(system_message)
    logger.log_json(metadata, "metadata.json")

    # Save Browserbase session ID if available
    if bb_session_id is not None:
        logger.write_to_file(bb_session_id, "BB_session_id.txt")

    user_message_strs: list[str] = []
    for i, interaction in enumerate(interactions):
        screenshot = Image.fromarray(interaction.state["obs"]["screenshot"])

        screenshot_name = logger.log_screenshot(screenshot, step=i + 1)

        if "extra_element_properties" in interaction.state["obs"]:
            _ = logger.log_extra_element_properties(
                interaction.state["obs"]["extra_element_properties"], step=i + 1
            )

        last_model_inputs = interaction.state.get("agent_inputs", None)
        other_obs: dict[str, Any] | None = None
        if last_model_inputs is not None:
            other_obs = _strip_non_serializable({
                k: v for k, v in last_model_inputs.items()
                if k not in KEYS_TO_STRIP
            })
            user_message = stringify(last_model_inputs.get("user_message"))
        else:
            user_message = ""
        user_message_strs.append(user_message)

        user_message_name = logger.log_user_message(
            user_message_str=user_message, step=i + 1
        )

        if interaction is not None and interaction.action is not None:
            action = {
                "action_str": interaction.action.get("action_str"),
                "action_description": interaction.action.get(
                    "action_description"
                ),
            }
            ao = interaction.action["action_output"].model_dump()
            ao["action_name"] = interaction.action["action_output"].action.name
            if ao["action_name"] == "click":
                bid = ao["action"]["bid"]
                extra_props = interaction.state["obs"].get(
                    "extra_element_properties", {}
                )
                axtree = interaction.state["obs"].get("axtree_object", {})
                ao["action"]["bbox"] = extra_props.get(bid, {}).get(
                    "bbox", None
                )
                # Only get node properties if axtree is populated (not empty for visual agents)
                ao["action"]["node_properties"] = (
                    get_node_properties(bid, axtree) if axtree else None
                )
            action["action_output"] = ao
        else:
            action = {
                "action_str": None,
                "action_description": None,
                "action_output": {
                    "thought": None,
                    "action": None,
                    "action_name": None,
                },
            }

        trajectory[i + 1] = {
            "screenshot": screenshot_name,
            "user_message": user_message_name,
            "other_obs": other_obs,
            "action": action,
            "error": interaction.error,
            "action_timestamp": interaction.action_timestamp,
            "raw_output": getattr(interaction, "raw_output", None),
        }

    logger.log_json(trajectory, "trajectory.json")

    elements = [
        ft.H1(f"{stringify(metadata.get('eps_name'))} ({task_type})"),
        ft.H2("Instruction"),
        ft.I(instruction),
        ft.H2("Trajectory"),
    ]

    for k, traj_step in trajectory.items():
        action = traj_step["action"]
        raw_output = traj_step.get("raw_output", None)

        ts = traj_step.get("action_timestamp")
        if ts is None:
            action_timestamp = "N/A"
        else:
            try:
                action_timestamp = datetime.fromtimestamp(ts).strftime(
                    "%Y-%m-%d | %H:%M:%S.%f"
                )
            except Exception:
                action_timestamp = "N/A"

        action_output_pretty = json.dumps(
            action.get("action_output", {}), ensure_ascii=False, indent=2
        )

        elements.extend(
            [
                ft.Div(
                    ft.Card(
                        ft.Div(
                            ft.Div(
                                ft.Img(src=f"images/{traj_step['screenshot']}"),
                                cls="col-xs-6",
                            ),
                            ft.Br(),
                            ft.Div(
                                ft.H6("Thought"),
                                ft.P(
                                    stringify(
                                        action.get("action_output", {}).get(
                                            "thought"
                                        )
                                    )
                                ),
                                ft.H6("Action"),
                                ft.P(stringify(action.get("action_str"))),
                                ft.H6("Action Description"),
                                ft.P(
                                    stringify(action.get("action_description"))
                                ),
                                cls="col-xs-6",
                            ),
                            cls="row",
                        ),
                        header=ft.Div(
                            f"Step {k}",
                            ft.Code(action_timestamp, style="float: right;"),
                        ),
                        footer=ft.Div(
                            ft.Details(
                                ft.Summary("System Message"),
                                ft.Pre(ft.Code(system_message)),
                            ),
                            ft.Details(
                                ft.Summary("User Message"),
                                ft.Pre(
                                    ft.Code(stringify(user_message_strs[k - 1]))
                                ),
                            ),
                            ft.Details(
                                ft.Summary("LLM Output (structured)"),
                                ft.Pre(ft.Code(action_output_pretty)),
                            ),
                            ft.Details(
                                ft.Summary("LLM Raw Output"),
                                ft.Pre(ft.Code(stringify(raw_output))),
                            ),
                        ),
                    ),
                ),
            ]
        )

    logger.write_to_file(ft.to_xml(create_page(elements)), "trajectory.html")
