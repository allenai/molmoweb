import asyncio
import json
import logging
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Literal

import fasthtml.common as ft
from PIL import Image

from agent.actions import ActionOutput, AxtreeActionOutput

SYSTEM_MESSAGE = "molmo_web_think"
from benchmarks.judges.webvoyager_judge import get_verdict
from benchmarks.judges.utils import OpenaiEngine, encode_image
from benchmarks.judges.webjudge_online_mind2web import (
    WebJudge_Online_Mind2Web_eval,
)
from benchmarks.traj_logging import log_episode
from utils.envs.browser_env import BrowserEnv
from utils.eval_utils.episode import Episode
from utils.eval_utils.episode_logger import LocalEpisodeLogger
from utils.vis_utils.html import create_page, create_table, save_html

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MIN_REMAINING_TO_RERUN = 20
MINIMUM_COMPLETED_TRAJECTORIES = 0.90
DEFAULT_MAX_RERUNS = 5


def read_trajectory(sample_dir: str):
    trajectory = json.load(
        open(os.path.join(sample_dir, "trajectory.json"), "r")
    )
    screenshots = []
    last_answer = "No answer"
    last_msg = "No answer"  # human traj doesn't use [ANSWER] tag, but we need to pass answers to webvoyager judge.
    last_thought = "No answer"
    for i, traj_step in trajectory.items():
        with Image.open(
            os.path.join(sample_dir, "images", traj_step["screenshot"])
        ) as img:
            screenshots.append(img.copy())

        action = traj_step["action"]["action_output"]["action"]
        if action is not None and "msg" in action:
            if "[ANSWER]" in action["msg"]:
                last_answer = action["msg"][
                    9:
                ]  # sometimes there are multiple answers
            elif "[EXIT]" not in action["msg"]:
                last_msg = action["msg"]

        # Track the last non-empty thought as a fallback
        thought = traj_step["action"]["action_output"].get("thought", "")
        if thought:
            last_thought = thought

    final_answer = (
        last_answer
        if last_answer != "No answer"
        else (last_msg if last_msg != "No answer" else last_thought)
    )
    return screenshots, final_answer


def run_trajectory_in_subprocess(kwargs: dict, timeout: float | None) -> str:
    def target(queue):
        try:
            result = get_trajectory(**kwargs)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(queue,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    else:
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result


def get_trajectory(
    sample: dict[str, str],
    outdir: str,
    agent_type: Literal[
        "gemini_cua",
        "gemini_axtree",
        "gpt_axtree",
        "molmoweb",
    ],
    inference_mode: Literal["local", "fastapi", "modal", "native", None] = "native",
    endpoint_or_checkpoint: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
    max_steps: int = 30,
    step_timeout_in_s: float = 120,
    llm_response_format: type[ActionOutput] = None,
    env_type: Literal["browserbase", "simple"] = "simple",
    max_past_steps: int = 3,
    max_past_images: int = 0,
    sampling_temperature: float = 0.7,
    sampling_top_p: float = 0.8,
):
    print(f"Starting task {sample['id']} (env={env_type})")

    start_url = sample.get("start_url", "about:blank")
    need_axtree = agent_type in ("gpt_axtree", "gemini_axtree")
    native_polyfill = sample.get("web_name", "").lower() in ("espn", "amazon")
    is_om2w = sample.get("task_type", "").startswith("online_mind2web")

    try:
        if env_type == "simple":
            from utils.envs import SimpleEnv
            env = SimpleEnv(
                start_url=start_url,
                goal=sample["prompt"],
                extract_axtree=need_axtree,
            )
        else:
            if native_polyfill:
                print(f"{sample.get('web_name')} detected -- using native polyfill for {sample['id']}")
            if is_om2w:
                print(f"om2w task -- using robust navigation for {sample['id']}")
            from utils.envs import BrowserbaseEnv
            env = BrowserbaseEnv(
                start_url=start_url,
                goal=sample["prompt"],
                extract_axtree=need_axtree,
                native_polyfill=native_polyfill,
                robust_navigation=is_om2w,
            )
    except Exception as e:
        print(f"Error in env creation: {str(e)}")
        return sample["id"]

    if llm_response_format is None:
        if agent_type in ("gpt_axtree", "gemini_axtree"):
            llm_response_format = AxtreeActionOutput

    try:
        if agent_type == "gemini_cua":
            from agent.gemini_cua import GeminiCUAgent
            agent = GeminiCUAgent()
        elif agent_type == "gpt_axtree":
            from agent.gpt_axtree_agent import GPTAxtreeAgent
            agent = GPTAxtreeAgent(
                llm_response_format=llm_response_format,
            )
        elif agent_type == "gemini_axtree":
            from agent.gemini_axtree_agent import GeminiAxtreeAgent
            agent = GeminiAxtreeAgent()
        elif agent_type == "molmoweb":
            from agent.multimodal_agent import MultimodalAgent
            assert endpoint_or_checkpoint is not None
            agent = MultimodalAgent(
                endpoint_or_checkpoint=endpoint_or_checkpoint,
                inference_mode=inference_mode,
                device=device,
                api_key=api_key,
                system_message=SYSTEM_MESSAGE,
                max_past_steps=max_past_steps,
                max_past_images=max_past_images,
                sampling_temperature=sampling_temperature,
                sampling_top_p=sampling_top_p,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    except Exception as e:
        print(f"Error in agent initialization: {str(e)}")
        env.close()
        return sample["id"]

    # Run episode
    episode = Episode(
        env=env,
        agent=agent,
        eps_name=sample["id"],
    )
    bb_session_id = None  # Initialize here

    try:
        interactions, metadata = episode.run_episode(max_steps=max_steps)
        
        # Safety check: ensure interactions list is not empty
        if not interactions:
            print(f"⚠️ Episode returned empty interactions list for {sample['id']}")
            interactions = episode.interactions if hasattr(episode, 'interactions') and episode.interactions else []
            if not interactions:
                print(f"❌ No interactions to log for {sample['id']}")
                env.close()
                return None
        
        if hasattr(env, "bb_session") and env.bb_session is not None:
            bb_session_id = env.bb_session.id

        # Determine output directory based on whether there was an error
        has_error = interactions and interactions[-1].error is not None
        episode_outdir = os.path.join(outdir, "errors") if has_error else outdir
        
        # print(f"💾 Logging trajectory for {sample['id']} to {episode_outdir} ({len(interactions)} interactions)")
        log_episode(
            interactions=interactions,
            metadata=metadata,
            system_message=SYSTEM_MESSAGE,
            outdir=(
                os.path.join(outdir, "errors")
                if interactions[-1].error is not None
                else outdir
            ),
            instruction=sample["prompt"],
            task_type=sample["task_type"],
            bb_session_id=bb_session_id,
        )
        print(f"💾 Successfully logged trajectory for {sample['id']}")

        if has_error:
            print(f"⚠️ Episode ended with error: {interactions[-1].error}")
            env.close()
            return None

    except Exception as e:
        print(f"❌ Error in running / logging episode: {str(e)}")

        if bb_session_id is None and hasattr(env, "bb_session") and env.bb_session is not None:
            bb_session_id = env.bb_session.id

        # Try to log whatever interactions we have, even if there was an error
        interactions_to_log = episode.interactions if hasattr(episode, 'interactions') and episode.interactions else []
        metadata_to_log = episode.metadata if hasattr(episode, 'metadata') and episode.metadata else {"goal": sample.get("prompt", ""), "eps_name": sample["id"]}
        
        if interactions_to_log:
            print(f"💾 Logging partial trajectory for {sample['id']} to errors directory ({len(interactions_to_log)} interactions)")
            try:
                log_episode(
                    interactions=interactions_to_log,
                    metadata=metadata_to_log,
                    system_message=SYSTEM_MESSAGE,
                    outdir=os.path.join(outdir, "errors"),
                    instruction=sample["prompt"],
                    task_type=sample["task_type"],
                    bb_session_id=bb_session_id,
                )
                print(f"✅ Successfully logged partial trajectory for {sample['id']}")
            except Exception as log_error:
                print(f"❌ Failed to log trajectory for {sample['id']}: {str(log_error)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ No interactions to log for {sample['id']} after error")

        env.close()
        return None

    env.close()
    return sample["id"]


def get_trajectories(
    samples: list[dict],  # list of samples in standard format
    results_dir: str,  # where to save results for this eval run
    agent_type: Literal[
        "gemini_cua",
        "gemini_axtree",
        "gpt_axtree",
        "molmoweb",
    ],
    inference_mode: Literal[
        "local", "fastapi", "modal", "native"
    ] = None,
    endpoint_or_checkpoint: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
    num_workers: int = 5,  # set to 0 for evaluating sequentially in the current process
    traj_timeout_in_s: float | None = 1800,
    step_timeout_in_s: float = 120,
    llm_response_format: type[ActionOutput] = None,
    max_steps: int = 30,
    env_type: Literal["browserbase", "simple"] = "simple",
    max_past_steps: int = 3,
    max_past_images: int = 0,
    sampling_temperature: float = 0.7,
    sampling_top_p: float = 0.8,
    min_remaining_to_rerun: int = DEFAULT_MIN_REMAINING_TO_RERUN,
    max_reruns: int = DEFAULT_MAX_RERUNS,
    _rerun_count: int = 0,
):
    os.makedirs(results_dir, exist_ok=True)

    futures = []
    if num_workers == 0:
        for i, sample in enumerate(samples):
            sample_results_dir = os.path.join(results_dir, sample["id"])
            trajectory_path = os.path.join(
                sample_results_dir, "trajectory.json"
            )
            if os.path.exists(trajectory_path):
                print(f"{trajectory_path} already exists; skipping")
                continue

            try:
                result = get_trajectory(
                    sample=sample,
                    outdir=sample_results_dir,
                    agent_type=agent_type,
                    inference_mode=inference_mode,
                    endpoint_or_checkpoint=endpoint_or_checkpoint,
                    device=device,
                    api_key=api_key,
                    max_steps=max_steps,
                    step_timeout_in_s=step_timeout_in_s,
                    llm_response_format=llm_response_format,
                    env_type=env_type,
                    max_past_steps=max_past_steps,
                    max_past_images=max_past_images,
                    sampling_temperature=sampling_temperature,
                    sampling_top_p=sampling_top_p,
                )
            except Exception as e:
                print(f"❌ Local task {sample['id']} failed: {str(e)}")
                continue
    else:
        assert (
            inference_mode != "local"
        ), "inference_mode='local' does not support multiprocess evaluation. Either use num_workers==0 OR use 'modal' or 'fastapi' inference modes"

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            future2sample = dict()
            for sample in samples:
                sample_results_dir = os.path.join(results_dir, sample["id"])
                trajectory_path = os.path.join(
                    sample_results_dir, "trajectory.json"
                )
                if os.path.exists(trajectory_path):
                    print(f"⚠️ {trajectory_path} already exists; skipping")
                    continue

                kwargs = dict(
                    sample=sample,
                    outdir=sample_results_dir,
                    agent_type=agent_type,
                    inference_mode=inference_mode,
                    endpoint_or_checkpoint=endpoint_or_checkpoint,
                    device=device,
                    api_key=api_key,
                    max_steps=max_steps,
                    step_timeout_in_s=step_timeout_in_s,
                    llm_response_format=llm_response_format,
                    env_type=env_type,
                    max_past_steps=max_past_steps,
                    max_past_images=max_past_images,
                    sampling_temperature=sampling_temperature,
                    sampling_top_p=sampling_top_p,
                )

                # Outer pool submits an inner single-process run with its own timeout
                future = executor.submit(
                    run_trajectory_in_subprocess, kwargs, traj_timeout_in_s
                )
                futures.append(future)
                future2sample[future] = sample

            num_samples = len(futures)
            print(f"💬 Added {num_samples} to the evaluation queue")
            num_eps_finished = 0

            try:
                for i, future in enumerate(as_completed(futures)):
                    sample = future2sample[future]
                    try:
                        result = future.result()
                        num_eps_finished += 1 if result is not None else 0
                        print(
                            f"✅ Attempted {i+1}th of {num_samples} ({num_eps_finished} completed successfully)"
                        )
                    except Exception as e:
                        print(
                            f"❌ Task {i+1}/{num_samples} encountered an error.\n\nError:\n{str(e)}\n\nTask Details:\n{sample}---"
                        )
                    finally:
                        if not future.done():
                            future.cancel()
            except Exception as e:
                print(
                    f"[Completed {num_eps_finished}/{len(futures)} episodes] - Error: {str(e)}"
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            num_remaining = len(futures) - num_eps_finished
            total_completed = sum(
                1 for s in samples
                if os.path.exists(os.path.join(results_dir, s["id"], "trajectory.json"))
            )
            completion_rate = total_completed / len(samples) if len(samples) > 0 else 1.0
            should_rerun = (
                _rerun_count < max_reruns
                and (num_remaining >= min_remaining_to_rerun or completion_rate < MINIMUM_COMPLETED_TRAJECTORIES)
            )
            if should_rerun:
                print(
                    f"Rerunning get_trajectories (rerun {_rerun_count + 1}/{max_reruns}) "
                    f"due to {num_remaining} remaining, {completion_rate:.1%} completion rate."
                )
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
                    llm_response_format=llm_response_format,
                    max_steps=max_steps,
                    env_type=env_type,
                    max_past_steps=max_past_steps,
                    max_past_images=max_past_images,
                    sampling_temperature=sampling_temperature,
                    sampling_top_p=sampling_top_p,
                    min_remaining_to_rerun=min_remaining_to_rerun,
                    max_reruns=max_reruns,
                    _rerun_count=_rerun_count + 1,
                )


@dataclass
class EvaluatedSample:
    id: str
    task: str
    status: Literal["COMPLETE", "MISSING"]
    answer: str | None = None
    last_screenshot: str | None = None
    verdict_thought: str | None = None
    verdict_judgement: Literal["SUCCESS", "FAILURE"] | None = None


def judge_trajectory(sample, exp_dir, judge_type, score_threshold=3):
    eps_name = sample["id"]
    eps_dir = os.path.join(exp_dir, eps_name)
    if not os.path.exists(eps_dir):
        print(f"🫥 Directory {eps_dir} missing")
        return EvaluatedSample(
            id=eps_name, task=sample["prompt"], status="MISSING"
        )
    try:
        screenshots, answer = read_trajectory(eps_dir)
        verdict_path = os.path.join(eps_dir, f"{judge_type}_verdict.json")
        if os.path.exists(verdict_path):
            with open(verdict_path, "r") as f:
                verdict = json.load(f)
            if "verdict" in verdict and verdict["verdict"] is not None:
                return EvaluatedSample(
                    id=eps_name,
                    task=sample["prompt"],
                    status="COMPLETE",
                    last_screenshot=f"{eps_name}/images/screenshot_{str(len(screenshots)).zfill(3)}.png",
                    answer=answer,
                    verdict_thought=verdict["thought"],
                    verdict_judgement=verdict["verdict"],
                )

        if judge_type == "webvoyager":
            verdict = get_verdict(
                task=sample["prompt"], answer=answer, screenshots=screenshots
            )
            LocalEpisodeLogger(eps_dir).log_json(
                data=verdict.model_dump(), fname=f"{judge_type}_verdict.json"
            )
            return EvaluatedSample(
                id=eps_name,
                task=sample["prompt"],
                status="COMPLETE",
                last_screenshot=f"{eps_name}/images/screenshot_{str(len(screenshots)).zfill(3)}.png",
                answer=answer,
                verdict_thought=verdict.thought,
                verdict_judgement=verdict.verdict,
            )
        elif judge_type == "webjudge_online_mind2web":
            images_path = [
                os.path.join(
                    eps_dir, "images", f"screenshot_{str(i+1).zfill(3)}.png"
                )
                for i in range(len(screenshots))
            ]

            last_actions = []
            try:
                trajectory = json.load(
                    open(os.path.join(eps_dir, "trajectory.json"), "r")
                )
                for _, step in trajectory.items():
                    if "action" in step and "action_str" in step["action"]:
                        last_actions.append(step["action"]["action_str"])
            except Exception:
                pass

            model = OpenaiEngine(
                model="o4-mini", api_key=os.getenv("OPENAI_API_KEY")
            )

            messages, text, system_msg, record, key_points = asyncio.run(
                WebJudge_Online_Mind2Web_eval(
                    task=sample["prompt"],
                    last_actions=last_actions,
                    images_path=images_path,
                    model=model,
                    score_threshold=score_threshold,
                )
            )

            response = model.generate(messages)[0]
            try:
                success = "success" in response.lower().split('status:')[1]
            except Exception:
                success = False
            thought = f"[OnlineMind2Web] Key points:\n{key_points}\n\nRecords: {record}\n\nFinal response: {response}"

            verdict = {
                "thought": thought,
                "verdict": "SUCCESS" if success else "FAILURE",
            }
            LocalEpisodeLogger(eps_dir).log_json(
                data=verdict, fname=f"{judge_type}_verdict.json"
            )

            return EvaluatedSample(
                id=eps_name,
                task=sample["prompt"],
                status="COMPLETE",
                last_screenshot=f"{eps_name}/images/screenshot_{str(len(screenshots)).zfill(3)}.png",
                answer=answer,
                verdict_thought=thought,
                verdict_judgement=verdict["verdict"],
            )
        elif judge_type == "deepshop_judge":
            from benchmarks.judges.deepshop_judge import get_verdict_deepshop
            verdict = get_verdict_deepshop(
                task=sample["prompt"],
                answer=answer,
                screenshots=screenshots,
            )
            LocalEpisodeLogger(eps_dir).log_json(
                data=verdict.model_dump(), fname=f"{judge_type}_verdict.json"
            )
            return EvaluatedSample(
                id=eps_name,
                task=sample["prompt"],
                status="COMPLETE",
                last_screenshot=f"{eps_name}/images/screenshot_{str(len(screenshots)).zfill(3)}.png",
                answer=answer,
                verdict_thought=verdict.thought,
                verdict_judgement=verdict.verdict,
            )
        else:
            raise ValueError(f"Unsupported judge_type: {judge_type}")

    except Exception as e:
        LocalEpisodeLogger(eps_dir).log_json(
            data={"error": str(e)}, fname=f"{judge_type}_verdict_error.json"
        )
        return EvaluatedSample(
            id=eps_name, task=sample["prompt"], status="MISSING"
        )


def judge_trajectories(
    samples: list[dict],
    results_dir: str,  # where to save results for this eval run
    judge_type: Literal["webvoyager", "webjudge_online_mind2web", "deepshop_judge"] = "webvoyager",
    num_workers: int = 5,
    grouping_mode: str | None = None,
):
    os.makedirs(results_dir, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx
    ) as executor:
        futures = []
        future2sample = dict()
        for sample in samples:
            future = executor.submit(
                judge_trajectory, sample, results_dir, judge_type
            )
            futures.append(future)
            future2sample[future] = sample

        records: list[EvaluatedSample] = []
        num_samples = len(futures)
        print(f"💬 Added {num_samples} to the evaluation queue")
        for i, future in enumerate(as_completed(futures)):
            sample = future2sample[future]
            try:
                res = future.result(timeout=600)
                records.append(res)
                print(
                    f"✅ Completed {i+1}/{num_samples}: {res.verdict_judgement}"
                )
            except FutureTimeoutError:
                print(
                    f"⏰ Task {i+1}/{num_samples} timed out.\n\nTask Details:\n{sample}"
                )
            except Exception as e:
                print(
                    f"❌ Task {i+1}/{num_samples} encountered an error.\n\nError:\n{str(e)}\n\nTask Details:\n{sample}---"
                )
            finally:
                if not future.done():
                    future.cancel()

    # Create mapping from sample id to level for online_mind2web grouping
    id_to_level = {}
    if grouping_mode == "online_mind2web":
        id_to_level = {
            sample["id"]: sample["level"]
            for sample in samples
            if "level" in sample
        }

    # Create mapping from sample id to difficulty for deepshop grouping
    id_to_difficulty = {}
    if grouping_mode == "deepshop":
        id_to_difficulty = {
            sample["id"]: sample["difficulty"]
            for sample in samples
            if "difficulty" in sample
        }

    # Create mapping from sample id to paper dimensions for deepshop_paper grouping
    # Each task can contribute to multiple dimensions (product_attribute, search_filter, sorting_preference) + task_success (overall)
    id_to_dimensions = {}
    if grouping_mode == "deepshop_paper":
        for sample in samples:
            dims = []
            if sample.get("has_attribute"):
                dims.append("product_attribute")
            if sample.get("has_filter"):
                dims.append("search_filter")
            if sample.get("has_sort"):
                dims.append("sorting_preference")
            id_to_dimensions[sample["id"]] = dims

    # Count steps for each record by reading trajectory.json
    step_counts = {}
    for record in records:
        if record.status == "COMPLETE":
            traj_path = os.path.join(results_dir, record.id, "trajectory.json")
            try:
                with open(traj_path, "r") as f:
                    trajectory = json.load(f)
                step_counts[record.id] = len(trajectory)
            except Exception:
                pass

    # Compute stats
    counter = Counter()
    website_counters = dict()
    website_steps = (
        dict()
    )  # steps per group: {group_id: {"all": [...], "success": [...]}}
    all_steps_list = []
    success_steps_list = []

    # Create mapping from sample id to web_name for website grouping
    id_to_web_name = {
        sample["id"]: sample.get("web_name")
        for sample in samples
        if "web_name" in sample
    }

    for record in records:
        if grouping_mode == "online_mind2web" and record.id in id_to_level:
            # Group by level (easy, medium, hard)
            id = id_to_level[record.id]
        elif grouping_mode == "deepshop" and record.id in id_to_difficulty:
            # Group by difficulty (easy, medium, hard)
            id = id_to_difficulty[record.id]
        elif grouping_mode == "deepshop_paper":
            # Paper Table 2: product_attribute, search_filter, sorting_preference, task_success (overall)
            # Each record contributes to applicable dimensions plus task_success
            id = None  # use group_ids instead
        elif grouping_mode == "website" and record.id in id_to_web_name:
            # Group by website name (e.g., "Allrecipes", "Amazon", etc.)
            id = id_to_web_name[record.id]
        else:
            # Default grouping: extract website ID from record.id
            parts = record.id.split("_")
            id = "_".join(parts[:-1]) if len(parts) > 1 else record.id

        if grouping_mode == "deepshop_paper":
            group_ids = id_to_dimensions.get(record.id, []) + ["task_success"]
        else:
            group_ids = [id]

        label = (
            "MISSING"
            if record.status == "MISSING"
            else record.verdict_judgement
        )

        counter[label] += 1

        for gid in group_ids:
            if gid not in website_counters:
                website_counters[gid] = Counter()
                website_steps[gid] = {"all": [], "success": []}
            website_counters[gid][label] += 1

            # Track steps (once per record for all groups it belongs to)
            if record.id in step_counts:
                steps = step_counts[record.id]
                if gid == group_ids[0]:  # only add to all_steps_list once per record
                    all_steps_list.append(steps)
                website_steps[gid]["all"].append(steps)
                if record.verdict_judgement == "SUCCESS":
                    if gid == group_ids[0]:
                        success_steps_list.append(steps)
                    website_steps[gid]["success"].append(steps)

    accuracy = dict()
    precision = dict()
    avg_steps = dict()
    avg_steps_success = dict()

    correct = counter["SUCCESS"]
    attempted = correct + counter["FAILURE"]
    all = attempted + counter["MISSING"]
    accuracy["all"] = round(correct * 100 / all, 2) if all > 0 else -1
    precision["all"] = (
        round(correct * 100 / attempted, 2) if attempted > 0 else -1
    )
    avg_steps["all"] = (
        round(sum(all_steps_list) / len(all_steps_list), 2)
        if all_steps_list
        else -1
    )
    avg_steps_success["all"] = (
        round(sum(success_steps_list) / len(success_steps_list), 2)
        if success_steps_list
        else -1
    )

    for id in website_counters:
        correct = website_counters[id]["SUCCESS"]
        attempted = correct + website_counters[id]["FAILURE"]
        all = attempted + website_counters[id]["MISSING"]
        accuracy[id] = round(correct * 100 / all, 2) if all > 0 else -1
        precision[id] = (
            round(correct * 100 / attempted, 2) if attempted > 0 else -1
        )
        grp_all = website_steps[id]["all"]
        grp_success = website_steps[id]["success"]
        avg_steps[id] = round(sum(grp_all) / len(grp_all), 2) if grp_all else -1
        avg_steps_success[id] = (
            round(sum(grp_success) / len(grp_success), 2) if grp_success else -1
        )

    columns = [
        "id",
        "missing",
        "success",
        "failure",
        "accuracy(all)",
        "accuracy(completed)",
        "steps(all)",
        "steps(success)",
    ]
    rows = [
        {
            "id": "ALL WEBSITES",
            "missing": counter["MISSING"],
            "success": counter["SUCCESS"],
            "failure": counter["FAILURE"],
            "accuracy(all)": accuracy["all"],
            "accuracy(completed)": precision["all"],
            "steps(all)": avg_steps["all"],
            "steps(success)": avg_steps_success["all"],
        }
    ]

    # Define category orders for specific grouping modes
    category_orders = {
        "online_mind2web": ["easy", "medium", "hard"],
        "deepshop": ["easy", "medium", "hard"],
        "deepshop_paper": ["product_attribute", "search_filter", "sorting_preference", "task_success"],
    }

    # Use predefined order if available, otherwise use all keys
    category_order = category_orders.get(grouping_mode)
    ids_to_iterate = (
        category_order if category_order else sorted(website_counters.keys())
    )

    for id in ids_to_iterate:
        if id in website_counters:
            rows.append(
                {
                    "id": id,
                    "missing": website_counters[id]["MISSING"],
                    "success": website_counters[id]["SUCCESS"],
                    "failure": website_counters[id]["FAILURE"],
                    "accuracy(all)": accuracy[id],
                    "accuracy(completed)": precision[id],
                    "steps(all)": avg_steps[id],
                    "steps(success)": avg_steps_success[id],
                }
            )

    table = create_table(columns=columns, rows=rows)

    elements = [
        ft.Details(
            ft.Summary(ft.B("Summary Stats"), role="button"),
            table,
        ),
        ft.Hr(),
        ft.H3("Verdicts"),
    ]
    for record in records:
        elements.extend(
            [
                ft.Card(
                    ft.Div(ft.I(record.task)),
                    (
                        ft.Div(
                            ft.Div(
                                ft.Img(src=record.last_screenshot),
                                cls="col-xs-6",
                            ),
                            ft.Div(
                                ft.H6("Answer"),
                                ft.P(record.answer),
                                ft.H6("Judgement"),
                                ft.Details(
                                    ft.Summary(record.verdict_judgement),
                                    ft.P(record.verdict_thought),
                                ),
                                cls="col-xs-6",
                            ),
                            cls="row",
                        )
                        if record.status == "COMPLETE"
                        else ""
                    ),
                    header=ft.B(record.id),
                    footer=ft.A(
                        "Trajectory",
                        href=f"{record.id}/trajectory.html",
                    ),
                )
            ]
        )

    save_html(
        elements, os.path.join(results_dir, f"!__{judge_type}_verdicts.html")
    )

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"Judge: {judge_type}")
    print(f"SUCCESS: {counter['SUCCESS']}  FAILURE: {counter['FAILURE']}  MISSING: {counter['MISSING']}")
    print(f"accuracy(all): {accuracy['all']}%")
    print(f"accuracy(completed): {precision['all']}%")
    print(sep)
