"""
Unified benchmark runner for all evaluation benchmarks.

Each benchmark is a different data loader that produces samples in a standard format:
    {"id": str, "prompt": str, "start_url": str, "task_type": str, ...}

Supported benchmarks: custom, deepshop, webvoyager, online_mind2web, webtailbench
"""
import fire
import json
import os
import random
import warnings
from pathlib import Path
from typing import Literal

# Suppress all Python warnings for cleaner benchmark logs.
warnings.filterwarnings("ignore")

from agent.actions import ActionOutput, AxtreeActionOutput

try:
    from requests.exceptions import RequestsDependencyWarning
except Exception:
    RequestsDependencyWarning = None

if RequestsDependencyWarning is not None:
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

JSONS_DIR = Path(__file__).parent / "jsons"

DEEPSHOP_JSONL_PATH = str(JSONS_DIR / "deepshop.jsonl")
WEBVOYAGER_DATA_PATH = str(JSONS_DIR / "webvoyager.jsonl")
ONLINE_MIND2WEB_DATA_PATH = str(JSONS_DIR / "online_mind2web.json")
WEBTAILBENCH_JSON_PATH = str(JSONS_DIR / "webtailbench.json")


def load_deepshop(data_path: str | Path) -> list[dict]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"deepshop data not found: {data_path}")

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)

            def _has(field: str) -> bool:
                v = task.get(field)
                if v is None:
                    return False
                s = str(v).strip().lower()
                return s not in ("none", "", "**")

            sample = {
                "id": task["id"],
                "prompt": task["ques"],
                "start_url": task["web"],
                "task_type": f"deepshop-{task['difficulty']}",
                "difficulty": task["difficulty"],
                "web_name": task.get("web_name"),
                "category": task.get("category"),
                "has_attribute": _has("attribute"),
                "has_filter": _has("filter"),
                "has_sort": _has("sort"),
            }
            samples.append(sample)

    return samples


def load_webvoyager(data_json: str) -> list[dict]:
    data = []
    with open(data_json, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    samples = []
    for d in data:
        sample = {
            "id": d["id"],
            "prompt": f"On {d['web']}: {d['ques']}",
            "start_url": "about:blank",
            "task_type": "webvoyager",
            "web_name": d.get("web_name", d["id"].split("--")[0]),
        }
        samples.append(sample)

    print(f"Loaded {len(samples)} WebVoyager samples")
    return samples


def load_custom(samples_json: str) -> list[dict]:
    """Load tasks from a plain JSON array of {id, prompt, task_type, ...} samples."""
    with open(samples_json, "r") as f:
        samples = json.load(f)
    return samples


def load_online_mind2web(data_json: str) -> list[dict]:
    with open(data_json) as f:
        tasks = json.load(f)

    samples = []
    for task in tasks:
        sample = {
            "id": task["task_id"],
            "prompt": task["confirmed_task"],
            "start_url": task["website"],
            "level": task["level"],
            "task_type": f"online_mind2web-{task['level']}",
        }
        samples.append(sample)

    return samples


def load_webtailbench(data_json: str) -> list[dict]:
    with open(data_json, "r") as f:
        tasks = json.load(f)

    samples = []
    for task in tasks:
        sample = {
            "id": f"{task['benchmark']}_{task['id']}",
            "prompt": task["task_summary"],
            "start_url": "about:blank",
            "task_type": f"webtailbench-{task['benchmark']}",
            "benchmark": task["benchmark"],
            "web_name": task["benchmark"],
        }
        samples.append(sample)

    return samples


BENCHMARK_DEFAULTS = {
    "custom": {
        "loader": load_custom,
        "data_path": None,
        "default_judge": "webvoyager",
        "grouping_mode": None,
    },
    "deepshop": {
        "loader": load_deepshop,
        "data_path": DEEPSHOP_JSONL_PATH,
        "default_judge": "deepshop_judge",
        "grouping_mode": "deepshop_paper",
    },
    "webvoyager": {
        "loader": load_webvoyager,
        "data_path": WEBVOYAGER_DATA_PATH,
        "default_judge": "webvoyager",
        "grouping_mode": "website",
    },
    "online_mind2web": {
        "loader": load_online_mind2web,
        "data_path": ONLINE_MIND2WEB_DATA_PATH,
        "default_judge": "webjudge_online_mind2web",
        "grouping_mode": "online_mind2web",
    },
    "webtailbench": {
        "loader": load_webtailbench,
        "data_path": WEBTAILBENCH_JSON_PATH,
        "default_judge": "webvoyager",
        "grouping_mode": "website",
    },
}


def _sample_phrase(n: int) -> str:
    return f"{n} sample" if n == 1 else f"{n} samples"


def _load_samples(benchmark: str, data_path: str | None = None) -> list[dict]:
    cfg = BENCHMARK_DEFAULTS[benchmark]
    path = data_path or cfg["data_path"]
    return cfg["loader"](path)


def run(
    results_dir: str,
    agent_type: Literal["gemini_cua", "gemini_axtree", "gpt_axtree", "molmoweb"],
    benchmark: Literal["custom", "deepshop", "webvoyager", "online_mind2web", "webtailbench"] = "custom",
    data_path: str | None = None,
    subset: str = "full",
    inference_mode: Literal["local", "fastapi", "modal", "native", None] = None,
    endpoint_or_checkpoint: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
    num_workers: int = 5,
    traj_timeout_in_s: float | None = 1800,
    step_timeout_in_s: float = 120,
    max_steps: int = 30,
    env_type: Literal["browserbase", "simple"] = "simple",
    llm_response_format: ActionOutput = None,
    seed: int = 123,
    max_past_steps: int = 10,
    max_past_images: int = 0,
    sampling_temperature: float = 0.7,
    sampling_top_p: float = 0.8,
):
    if llm_response_format is None:
        if agent_type in ["gpt_axtree", "gemini_axtree"]:
            llm_response_format = AxtreeActionOutput

    samples = _load_samples(benchmark, data_path)
    print(f"Loaded {_sample_phrase(len(samples))} for {benchmark} benchmark")

    random.seed(seed)
    random.shuffle(samples)

    if subset.startswith("range_"):
        _, start, end = subset.split("_")
        samples = samples[int(start):int(end)]
        print(f"Subset {subset}: running {_sample_phrase(len(samples))}")

    from benchmarks.evaluate import get_trajectories
    get_trajectories(
        samples=samples,
        results_dir=results_dir,
        agent_type=agent_type,
        inference_mode=inference_mode,
        endpoint_or_checkpoint=endpoint_or_checkpoint,
        device=device,
        api_key=api_key,
        num_workers=num_workers,
        traj_timeout_in_s=traj_timeout_in_s,
        step_timeout_in_s=step_timeout_in_s,
        max_steps=max_steps,
        env_type=env_type,
        llm_response_format=llm_response_format,
        max_past_steps=max_past_steps,
        max_past_images=max_past_images,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
    )


def judge(
    results_dir: str,
    benchmark: Literal["custom", "deepshop", "webvoyager", "online_mind2web", "webtailbench"] = "custom",
    data_path: str | None = None,
    judge_type: Literal[
        "webjudge_online_mind2web",
        "webvoyager",
        "deepshop_judge",
    ] | None = None,
    num_workers: int = 30,
    seed: int = 123,
    grouping_mode: str | None = None,
):
    cfg = BENCHMARK_DEFAULTS[benchmark]
    if judge_type is None:
        judge_type = cfg["default_judge"]
    if grouping_mode is None:
        grouping_mode = cfg["grouping_mode"]

    samples = _load_samples(benchmark, data_path)
    print(f"Loaded {_sample_phrase(len(samples))} for {benchmark} benchmark judging")

    random.seed(seed)
    random.shuffle(samples)

    from benchmarks.evaluate import judge_trajectories
    judge_trajectories(
        samples,
        results_dir,
        judge_type,
        num_workers,
        grouping_mode=grouping_mode,
    )


if __name__ == "__main__":
    fire.Fire()
