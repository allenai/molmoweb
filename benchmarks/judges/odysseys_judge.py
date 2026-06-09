"""Rubric-based judge for the Odysseys benchmark.

Adapted from the upstream Odysseys full-trajectory rubric judge so MolmoWeb
can score its own trajectory format directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import AsyncOpenAI

DEFAULT_MODEL = os.getenv("ODYSSEYS_JUDGE_MODEL", "gemini-3.1-flash-lite-preview")
DEFAULT_MAX_IMAGES = int(os.getenv("ODYSSEYS_JUDGE_MAX_IMAGES", "0"))
DEFAULT_MAX_STEPS = int(os.getenv("ODYSSEYS_JUDGE_MAX_STEPS", "0"))
FINAL_JUDGMENT_MAX_COMPLETION_TOKENS = 8192

FULL_TRAJ_JUDGMENT_SYSTEM = """You are an expert evaluator of web-navigation agent trajectories.

You will receive:
- The user task (for context).
- ONE specific rubric item with a requirement and a verification description.
- The agent's full action history (one line per step).
- Every screenshot from the trajectory, in chronological order.

Your goal is to decide whether this single rubric item is satisfied by the trajectory.

Evaluation rules:
- Judge ONLY the one rubric item you are given; ignore all other implicit requirements.
- Ground your judgment in what the screenshots and actions actually show. Do not invent state.
- Filtering / sorting / form requirements must be applied and confirmed to count as satisfied.
- If the agent was blocked (captcha, access denied, etc.) and therefore could not satisfy the rubric, report failure.

Respond in exactly this format:

Thoughts: <your reasoning, citing specific steps/screenshots>
Status: "success" or "failure"
"""


def _make_client(model: str) -> tuple[str, Any]:
    if model.lower().startswith("gemini"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key required for odysseys_rubric. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        return "gemini", genai.Client(api_key=api_key)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key required for odysseys_rubric when using a non-Gemini model. "
            "Set OPENAI_API_KEY."
        )
    return "openai", AsyncOpenAI(api_key=api_key)


def _normalize_rubrics(
    rubrics: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if rubrics is None:
        return []
    if isinstance(rubrics, dict):
        return [{"id": str(rid), **value} for rid, value in rubrics.items()]
    return [
        {
            "id": str(item.get("id", f"R{i+1}")),
            "requirement": item.get("requirement", ""),
            "verification": item.get("verification", ""),
        }
        for i, item in enumerate(rubrics)
    ]


def _load_molmoweb_trajectory(
    sample_dir: str | Path,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[dict[str, Any]]:
    sample_dir = Path(sample_dir)
    trajectory_path = sample_dir / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())

    steps = []
    for step_key in sorted(trajectory.keys(), key=lambda x: int(x)):
        step_num = int(step_key)
        if max_steps > 0 and step_num > max_steps:
            continue

        traj_step = trajectory[step_key]
        action = traj_step.get("action", {})
        action_output = action.get("action_output", {})
        action_payload = action_output.get("action", {}) or {}

        steps.append(
            {
                "step": step_num,
                "thought": str(action_output.get("thought", "")).strip(),
                "action_line": str(action.get("action_str", "")).strip(),
                "message": str(action_payload.get("msg", "")).strip(),
                "screenshot": sample_dir / "images" / traj_step["screenshot"],
            }
        )

    return steps


def _build_action_history(steps: list[dict[str, Any]]) -> str:
    lines = []
    for idx, step in enumerate(steps, start=1):
        parts = []
        if step["thought"]:
            parts.append(f"Thought: {step['thought']}")
        if step["action_line"]:
            parts.append(f"Action: {step['action_line']}")
        if step["message"] and "[EXIT]" not in step["message"]:
            parts.append(f"Message: {step['message']}")
        if parts:
            lines.append(f"{idx}. " + "\n".join(parts))
    return "\n".join(lines) if lines else "No actions recorded."


def _load_screenshot_assets(
    steps: list[dict[str, Any]], max_images: int = DEFAULT_MAX_IMAGES
) -> list[dict[str, Any]]:
    screenshot_assets = []
    screenshot_steps = steps[-max_images:] if max_images > 0 else steps
    for step in screenshot_steps:
        path = step["screenshot"]
        if not path.exists():
            continue
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        screenshot_assets.append(
            {
                "bytes": data,
                "mime": mime,
                "data_url": f"data:{mime};base64,{base64.b64encode(data).decode()}",
            }
        )
    return screenshot_assets


async def _evaluate_rubric(
    client: Any,
    backend: str,
    model: str,
    task: str,
    action_history: str,
    screenshot_assets: list[dict[str, Any]],
    screenshot_count: int,
    total_steps: int,
    rubric: dict[str, Any],
) -> dict[str, Any]:
    rubric_id = rubric.get("id", "?")
    rubric_lines = [
        f"Rubric ID: {rubric_id}",
        f"Requirement: {str(rubric.get('requirement', '')).strip()}",
    ]
    verification = str(rubric.get("verification", "")).strip()
    if verification:
        rubric_lines.append(f"Verification: {verification}")

    user_text = (
        f"User Task (context only): {task}\n\n"
        "Evaluate ONLY this rubric item:\n"
        + "\n".join(rubric_lines)
        + f"\n\nFull Action History:\n{action_history}\n\n"
        + f"Screenshots attached below: {screenshot_count} "
        + f"(trajectory had {total_steps} total step(s)).\n\n"
        + f"Decide whether the rubric ({rubric_id}) is satisfied. "
        + "Use the required 'Thoughts:' / 'Status:' format."
    )

    try:
        if backend == "gemini":
            response = await client.aio.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_text)]
                        + [
                            types.Part.from_bytes(
                                data=asset["bytes"], mime_type=asset["mime"]
                            )
                            for asset in screenshot_assets
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=FULL_TRAJ_JUDGMENT_SYSTEM,
                    max_output_tokens=FINAL_JUDGMENT_MAX_COMPLETION_TOKENS,
                ),
            )
            result_text = str(response.text or "").strip()
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FULL_TRAJ_JUDGMENT_SYSTEM},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}]
                        + [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": asset["data_url"],
                                    "detail": "high",
                                },
                            }
                            for asset in screenshot_assets
                        ],
                    },
                ],
                max_completion_tokens=FINAL_JUDGMENT_MAX_COMPLETION_TOKENS,
            )
            result_text = str(response.choices[0].message.content or "").strip()

        status_match = re.search(
            r'Status:\s*["\']?(success|failure)["\']?', result_text, re.IGNORECASE
        )
        thoughts_match = re.search(
            r"Thoughts:\s*(.+?)(?:Status:|$)", result_text, re.DOTALL
        )
        success = bool(status_match and status_match.group(1).lower() == "success")
        reasoning = (
            thoughts_match.group(1).strip()
            if thoughts_match
            else result_text.strip() or "Empty judge response."
        )
    except Exception as exc:
        success = False
        reasoning = f"Error judging rubric {rubric_id}: {exc}"

    return {
        "rubric_id": rubric_id,
        "requirement": rubric.get("requirement", ""),
        "verification": rubric.get("verification", ""),
        "score": 1 if success else 0,
        "success": success,
        "final_reasoning": reasoning,
    }


async def _judge_odysseys_async(
    task: str,
    rubrics: dict[str, Any] | list[dict[str, Any]] | None,
    sample_dir: str | Path,
    model: str = DEFAULT_MODEL,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    normalized_rubrics = _normalize_rubrics(rubrics)
    if not normalized_rubrics:
        raise ValueError("No rubrics found for Odysseys task.")

    steps = _load_molmoweb_trajectory(sample_dir, max_steps=max_steps)
    if not steps:
        raise ValueError("Empty trajectory - no steps recorded.")

    action_history = _build_action_history(steps)
    screenshot_assets = _load_screenshot_assets(steps, max_images=max_images)
    backend, client = _make_client(model)

    rubric_results = []
    for rubric in normalized_rubrics:
        result = await _evaluate_rubric(
            client=client,
            backend=backend,
            model=model,
            task=task,
            action_history=action_history,
            screenshot_assets=screenshot_assets,
            screenshot_count=len(screenshot_assets),
            total_steps=len(steps),
            rubric=rubric,
        )
        rubric_results.append(result)

    rubric_scores = {item["rubric_id"]: item["score"] for item in rubric_results}
    average = sum(rubric_scores.values()) / len(rubric_scores) if rubric_scores else 0.0
    perfect = bool(rubric_scores) and all(
        score == 1 for score in rubric_scores.values()
    )
    trajectory_efficiency = (average / len(steps)) if steps else 0.0

    return {
        "thought": "\n\n".join(
            f"{item['rubric_id']}: {'SUCCESS' if item['success'] else 'FAILURE'}\n{item['final_reasoning']}"
            for item in rubric_results
        ),
        "verdict": "SUCCESS" if perfect else "FAILURE",
        "judge_model": model,
        "num_steps": len(steps),
        "num_screenshots_sent": len(screenshot_assets),
        "rubric_scores": rubric_scores,
        "rubric_results": rubric_results,
        "average_rubric_score": round(average, 4),
        "perfect": perfect,
        "trajectory_efficiency": round(trajectory_efficiency, 6),
        "trajectory_efficiency_x100": round(trajectory_efficiency * 100, 4),
    }


def get_odysseys_rubric_verdict(
    task: str,
    rubrics: dict[str, Any] | list[dict[str, Any]] | None,
    sample_dir: str | Path,
    model: str = DEFAULT_MODEL,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    return asyncio.run(
        _judge_odysseys_async(
            task=task,
            rubrics=rubrics,
            sample_dir=sample_dir,
            model=model,
            max_images=max_images,
            max_steps=max_steps,
        )
    )
