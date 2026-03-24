"""
MolmoWeb client for inference.
"""

import asyncio
import os
import re
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

from agent.actions import ActionOutput, SendMsgToUser
from agent.multimodal_agent import MultimodalAgent
from inference.web_episode import State, Step, Trajectory
from utils.axtree import extract_axtree, flatten_axtree_to_str, EXTRACT_OBS_MAX_TRIES


def _check_browserbase_credentials() -> None:
    missing = []
    if not os.environ.get("BROWSERBASE_API_KEY", "").strip():
        missing.append("BROWSERBASE_API_KEY")
    if not os.environ.get("BROWSERBASE_PROJECT_ID", "").strip():
        missing.append("BROWSERBASE_PROJECT_ID")
    if missing:
        raise ValueError(
            f"Missing environment variable(s) for Browserbase: {', '.join(missing)}. "
            "Set them or use local=True to run with a local browser."
        )


def _run_one_query(
    endpoint: str,
    local: bool,
    query: str,
    max_steps: int,
    headless: bool = True,
) -> Trajectory:
    """Worker function for run_batch -- runs a single query in its own process."""
    client = MolmoWeb(
        endpoint=endpoint,
        local=local,
        keep_alive=False,
        headless=headless,
        verbose=False,
    )
    return client.run(query=query, max_steps=max_steps)


class MolmoWeb:
    """
    Client for running the Molmo web agent.

    When local=True, uses a local Chromium browser (no credentials needed).
    When local=False, uses Browserbase (requires BROWSERBASE_API_KEY and
    BROWSERBASE_PROJECT_ID environment variables).
    """

    VIEWPORT_WIDTH = 1280
    VIEWPORT_HEIGHT = 720

    def __init__(
        self,
        endpoint: str | None = None,
        local: bool = True,
        keep_alive: bool = True,
        headless: bool = True,
        verbose: bool = True,
    ):
        self.endpoint = endpoint or os.environ.get("MOLMOWEB_ENDPOINT")
        self.local = local
        self.keep_alive = keep_alive
        self.headless = headless
        self.verbose = verbose
        self.agent = self._create_agent() if self.endpoint else None
        self.env = None
        self.last_obs = None
        self._pw_event_loop: asyncio.AbstractEventLoop | None = None

    @contextmanager
    def _pw_context(self):
        """Set the asyncio running loop to Playwright's internal loop so that
        Playwright sync calls work across separate invocations.
        """
        saved = asyncio._get_running_loop()
        asyncio._set_running_loop(self._pw_event_loop)
        try:
            yield
        finally:
            asyncio._set_running_loop(saved)

    def _create_agent(self) -> MultimodalAgent:
        return MultimodalAgent(
            endpoint_or_checkpoint=self.endpoint,
            system_message="molmo_web_think",
            inference_mode="fastapi",
            max_past_steps=10,
            max_past_images=0,
        )

    def _create_env(self, start_url: str = "about:blank"):
        if self.local:
            from utils.envs import SimpleEnv

            return SimpleEnv(
                start_url=start_url,
                goal="",
                viewport_width=self.VIEWPORT_WIDTH,
                viewport_height=self.VIEWPORT_HEIGHT,
                extract_axtree=False,
                headless=self.headless,
            )

        _check_browserbase_credentials()
        from utils.envs import BrowserbaseEnv

        return BrowserbaseEnv(
            start_url=start_url,
            goal="",
            viewport_width=self.VIEWPORT_WIDTH,
            viewport_height=self.VIEWPORT_HEIGHT,
            extract_axtree=False,
        )

    def _get_state(self, obs: dict) -> State:
        page_index = int(obs["active_page_index"][0])
        return State(
            img=Image.fromarray(obs["screenshot"]),
            page_url=obs["open_pages_urls"][page_index],
            page_title=obs["open_pages_titles"][page_index],
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(Exception),
    )
    def _predict(self, obs: dict, query: str) -> dict:
        obs["goal"] = query
        _, action = self.agent.predict_action(obs)
        return action

    def _step_env(self, action: dict) -> dict:
        assert self.env is not None, "Environment not initialized"
        action_obj = action["action_output"].action
        return self.env.step(action_obj)

    def _run_one(self, obs: dict, query: str) -> tuple[dict | None, Step]:
        h, w = obs["screenshot"].shape[:2]
        if (w, h) != (self.VIEWPORT_WIDTH, self.VIEWPORT_HEIGHT):
            img = Image.fromarray(obs["screenshot"]).resize(
                (self.VIEWPORT_WIDTH, self.VIEWPORT_HEIGHT), Image.LANCZOS,
            )
            obs["screenshot"] = np.array(img)
        state = self._get_state(obs)
        try:
            action = self._predict(obs, query)
            prediction: ActionOutput = action["action_output"]
        except Exception as e:
            return None, Step(state=state, prediction=None, error=str(e))

        try:
            next_obs = self._step_env(action)
            self.last_obs = next_obs
        except Exception as e:
            return None, Step(state=state, prediction=prediction, error=str(e))

        return next_obs, Step(state=state, prediction=prediction, error=None)

    def _run_iters(self, obs: dict, query: str, max_steps: int) -> Trajectory:
        traj = Trajectory()
        curr_obs = obs
        for step_num in range(1, max_steps + 1):
            next_obs, step = self._run_one(curr_obs, query)
            traj.steps.append(step)

            if step.error is not None:
                if self.verbose:
                    print(f"[{datetime.now():%H:%M:%S}] Step {step_num:2d}: [error] {step.error}")
                return traj

            if self.verbose:
                print(f"[{datetime.now():%H:%M:%S}] Step {step_num:2d}: {step.prediction.action}")

            curr_obs = next_obs

            if isinstance(step.prediction.action, SendMsgToUser) and (
                step.prediction.action.msg.startswith("[EXIT]")
                or step.prediction.action.msg.startswith("[ANSWER]")
            ):
                return traj

        return traj

    def fresh_run(self, query: str, max_steps: int) -> Trajectory:
        with self._pw_context():
            self.last_obs = None
            self.env = self._create_env()
            self.agent.reset()
            obs, _ = self.env.reset()
            self._pw_event_loop = asyncio._get_running_loop()
            return self._run_iters(obs, query, max_steps)

    def continue_run(self, query: str, max_steps: int) -> Trajectory:
        with self._pw_context():
            if self.last_obs is None:
                raise ValueError("Cannot continue without a previous observation")
            self.agent.reset()
            obs = self.env._get_obs()
            return self._run_iters(obs, query, max_steps)

    def run(self, query: str, max_steps: int = 15) -> Trajectory:
        with self._pw_context():
            if self.env is None:
                traj = self.fresh_run(query, max_steps)
            else:
                traj = self.continue_run(query, max_steps)

            if not self.keep_alive:
                self.close()

            return traj

    def run_batch(
        self,
        queries: list[str],
        max_steps: int,
        max_workers: int = 4,
        headless: bool = True,
        output_dir: str | None = None,
    ) -> list[Trajectory]:
        """Run multiple queries in parallel (one process per query).

        Returns trajectories in the same order as queries.
        """
        if not queries:
            return []

        out = Path(output_dir) if output_dir is not None else Path("inference/htmls")
        out.mkdir(parents=True, exist_ok=True)

        workers = min(max_workers, len(queries))
        results: list[Trajectory | None] = [None] * len(queries)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx: dict = {}
            for i, query in enumerate(queries):
                f = executor.submit(
                    _run_one_query,
                    self.endpoint,
                    self.local,
                    query,
                    max_steps,
                    headless,
                )
                future_to_idx[f] = i

            with tqdm(total=len(queries), desc="Batch") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    traj = future.result()
                    results[idx] = traj

                    slug = re.sub(r"[^\w]+", "_", queries[idx])[:60].strip("_")
                    filename = f"{idx:03d}_{slug}.html"
                    traj.save_html(
                        output_path=str(out / filename),
                        query=queries[idx],
                    )

                    label = queries[idx][:60] + ("..." if len(queries[idx]) > 60 else "")
                    tqdm.write(f'  [{idx}] "{label}" ({len(traj.steps)} steps)')
                    pbar.update(1)

        return results

    def get_axtree(self, url: str | None = None, **flatten_kwargs) -> str:
        """Return the flattened axtree string for a page.

        If *url* is given the browser navigates there first (creating the
        environment if needed).  Without *url* the current page is used.
        Extra keyword arguments are forwarded to ``flatten_axtree_to_str``.
        """
        with self._pw_context():
            if url is not None:
                if self.env is None:
                    self.env = self._create_env(start_url=url)
                    self.env.reset(start_url=url)
                    self._pw_event_loop = asyncio._get_running_loop()
                else:
                    self.env.page.goto(url, wait_until="domcontentloaded")
                    try:
                        self.env.page.wait_for_load_state("networkidle", timeout=10_000)
                    except Exception:
                        pass
            elif self.env is None:
                raise ValueError("No url provided and environment not initialized")

            for attempt in reversed(range(EXTRACT_OBS_MAX_TRIES)):
                try:
                    axtree_obj, extra_props = extract_axtree(self.env.page, lenient=(attempt == 0))
                    break
                except Exception:
                    if attempt == 0:
                        raise

            return flatten_axtree_to_str(axtree_obj, extra_props, **flatten_kwargs)

    def close(self) -> None:
        with self._pw_context():
            if self.env is not None:
                self.env.close()
                self.env = None
