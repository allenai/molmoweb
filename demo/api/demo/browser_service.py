"""Browser service that manages BrowserBase sessions and executes actions."""

import asyncio
import base64
import logging
import os
import queue
import time
import random
import uuid
from io import BytesIO
from threading import Lock, Thread
from typing import Any, Callable, Optional

import numpy as np
from agent.actions import (
    BrowserNav,
    GeminiTypeTextAt,
    Goto,
    KeyboardPress,
    KeyboardType,
    MouseClick,
    MouseDragAndDrop,
    MouseMove,
    Noop,
    ReportInfeasible,
    Scroll,
    ScrollAt,
    SendMsgToUser,
)
from browserbase import Browserbase
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

from .agent_factory import create_agent
from .config import settings
from .mem_util import log_memory
from .session_manager import session_manager

logger = logging.getLogger(__name__)

# uuid for the process
_process_id = uuid.uuid4()
_active_threads = 0
_active_threads_lock = Lock()


def take_screenshot_fast(page) -> bytes:
    """
    Take a screenshot using CDP directly.
    Falls back to regular screenshot if CDP fails.
    """
    try:
        # Use CDP to capture screenshot without waiting for fonts
        cdp = page.context.new_cdp_session(page)
        result = cdp.send("Page.captureScreenshot", {"format": "png"})
        cdp.detach()
        return base64.b64decode(result["data"])
    except Exception as e:
        print(f"[WORKER] CDP screenshot failed, falling back to Playwright: {e}")
        # Fallback to regular screenshot with short timeout
        return page.screenshot(timeout=10000, animations="disabled")


def start_playwright_sync():
    """Start Playwright sync API, bypassing asyncio loop detection.

    In threaded environments, an asyncio loop may exist that triggers
    Playwright's async detection even though we're not actually in an
    async context. We clear the running loop state and set up a fresh
    event loop for this thread.
    """
    # Clear any running loop detection
    asyncio._set_running_loop(None)

    # Create a fresh event loop for this thread
    # This ensures Playwright has a clean loop to work with
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return sync_playwright().start()


def add_click_indicator(
    screenshot_base64: str, x: float, y: float, screenshot_shape: tuple
) -> str:
    """
    Add a red circle indicator at the click position on the screenshot.

    Args:
        screenshot_base64: Base64 encoded PNG screenshot
        x: X coordinate in pixels
        y: Y coordinate in pixels
        screenshot_shape: (height, width) of the original screenshot

    Returns:
        Base64 encoded PNG with click indicator
    """
    img_data = base64.b64decode(screenshot_base64)
    img = Image.open(BytesIO(img_data)).convert("RGBA")

    height, width = screenshot_shape[:2]
    x_pixel = x
    y_pixel = y

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    radius_outer = 25
    radius_inner = 15
    radius_dot = 5

    draw.ellipse(
        [
            x_pixel - radius_outer,
            y_pixel - radius_outer,
            x_pixel + radius_outer,
            y_pixel + radius_outer,
        ],
        outline=(239, 68, 68, 200),
        width=3,
    )

    draw.ellipse(
        [
            x_pixel - radius_inner,
            y_pixel - radius_inner,
            x_pixel + radius_inner,
            y_pixel + radius_inner,
        ],
        outline=(239, 68, 68, 255),
        width=2,
    )

    draw.ellipse(
        [
            x_pixel - radius_dot,
            y_pixel - radius_dot,
            x_pixel + radius_dot,
            y_pixel + radius_dot,
        ],
        fill=(239, 68, 68, 255),
    )

    result = Image.alpha_composite(img, overlay)

    buffer = BytesIO()
    result.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


_CC_PATTERN = r"credit.?card|card.?number|cc.?num|cvv|cvc|ccv|security.?code"


def _check_sensitive_field(page) -> str | None:
    """Return an error message if the focused element is a sensitive field, else None."""
    import re

    attrs = page.evaluate("""(() => {
        const el = document.activeElement;
        if (!el) return null;
        return {
            type: el.type || '',
            autocomplete: el.getAttribute('autocomplete') || '',
            name: el.getAttribute('name') || '',
            id: el.id || '',
        };
    })()""")
    if not attrs:
        return None
    if attrs["type"] == "password":
        return "Refused to type into a password field."
    autocomplete = attrs["autocomplete"].lower()
    if autocomplete.startswith("cc-"):
        return "Refused to type into a credit card field."
    name = attrs["name"].lower()
    el_id = attrs["id"].lower()
    if re.search(_CC_PATTERN, name) or re.search(_CC_PATTERN, el_id):
        return "Refused to type into a credit card field."
    return None


def execute_action_from_output(
    page, action_output, viewport_width: int = 1280, viewport_height: int = 720
) -> tuple[bool, str]:
    """
    Execute an action from the agent's ActionOutput on the Playwright page.

    Args:
        page: Playwright page
        action_output: ActionOutput object from agent
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height

    Returns:
        (success, error_message)
    """

    action = action_output.action

    try:
        if isinstance(action, MouseClick):
            if action.click_type == "double":
                page.mouse.dblclick(action.x, action.y)
            else:
                # randx randy for the click for 0.01 - 0.02 px
                # page.mouse.click(action.x + random.uniform(-0.01, 0.01), action.y + random.uniform(-0.01, 0.01))
                page.mouse.click(action.x, action.y)
            return True, ""

        elif isinstance(action, MouseMove):
            page.mouse.move(action.x, action.y, steps=3)
            return True, ""

        elif isinstance(action, MouseDragAndDrop):
            page.mouse.move(action.from_x, action.from_y, steps=3)
            page.mouse.down()
            page.mouse.move(action.to_x, action.to_y, steps=3)
            page.mouse.up()
            return True, ""

        elif isinstance(action, Scroll):
            page.mouse.wheel(action.delta_x, action.delta_y)
            return True, ""

        elif isinstance(action, ScrollAt):
            page.mouse.move(action.x, action.y, steps=3)
            page.mouse.wheel(action.dx, action.dy)
            return True, ""

        elif isinstance(action, KeyboardType):
            sensitive_err = _check_sensitive_field(page)
            if sensitive_err:
                return False, sensitive_err
            page.keyboard.type(action.text)
            return True, ""

        elif isinstance(action, KeyboardPress):
            sensitive_err = _check_sensitive_field(page)
            if sensitive_err:
                return False, sensitive_err
            page.keyboard.press(action.key)
            return True, ""

        elif isinstance(action, GeminiTypeTextAt):
            # Click at position, optionally clear, type text, optionally press Enter
            page.mouse.click(action.x, action.y)
            sensitive_err = _check_sensitive_field(page)
            if sensitive_err:
                return False, sensitive_err
            if action.clear_before_typing:
                page.keyboard.press("Control+a")
                page.keyboard.press("Backspace")
            page.keyboard.type(action.text)
            if action.press_enter:
                page.keyboard.press("Enter")
            return True, ""

        elif isinstance(action, Goto):
            page.goto(action.url, wait_until="domcontentloaded")
            return True, ""

        elif isinstance(action, BrowserNav):
            if action.nav_type == "go_back":
                page.go_back()
            elif action.nav_type == "new_tab":
                page.context.new_page()
            elif action.nav_type == "tab_focus":
                pages = page.context.pages
                if 0 <= action.index < len(pages):
                    pages[action.index].bring_to_front()
            return True, ""

        elif isinstance(action, Noop):
            # Wait for page load state (or up to 15s); fast sites finish sooner
            try:
                page.wait_for_load_state("load", timeout=15_000)
            except Exception:
                # Timeout or other: continue anyway (we waited as long as we could)
                pass
            return True, ""

        elif isinstance(action, (SendMsgToUser, ReportInfeasible)):
            # Terminal actions - no browser action needed
            return True, ""

        else:
            return False, f"Unknown action type: {type(action)}"

    except Exception as e:
        return False, str(e)


def build_observation(
    page,
    goal: str,
    last_action_error: str = "",
    screenshot_bytes: bytes = None,
) -> dict[str, Any]:
    """
    Build observation dict from the current page state.

    This matches the format expected by both molmoweb (MultimodalAgent) and GeminiCUAgent.

    Args:
        screenshot_bytes: Optional pre-captured screenshot. If None, takes a new one.
    """
    # Take screenshot if not provided
    if screenshot_bytes is None:
        screenshot_bytes = take_screenshot_fast(page)
    img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
    screenshot_np = np.array(img)

    # Get page info
    current_url = page.url
    page_title = page.title()

    # Get all open pages
    open_pages_titles = []
    open_pages_urls = []
    active_page_index = 0

    for i, p in enumerate(page.context.pages):
        try:
            open_pages_titles.append(p.title())
            open_pages_urls.append(p.url)
            if p == page:
                active_page_index = i
        except:
            open_pages_titles.append("Unknown")
            open_pages_urls.append("")

    return {
        "goal": goal,
        "screenshot": screenshot_np,
        "url": current_url,
        "open_pages_titles": open_pages_titles,
        "open_pages_urls": open_pages_urls,
        "active_page_index": [active_page_index],
        "last_action_error": last_action_error,
    }


def _obs_screenshot_to_base64(obs: dict[str, Any]) -> str:
    """Convert observation screenshot (numpy) to base64 PNG string for socket payloads."""
    buffer = BytesIO()
    Image.fromarray(obs["screenshot"]).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def browser_worker(
    command_queue: queue.Queue,
    result_queue: queue.Queue,
    browserbase_api_key: str,
    browserbase_project_id: str,
    start_url: str,
    goal: str,
    context: dict[str, Any] | None = None,
    session_id: Optional[str] = None,
):
    """
    Worker thread that runs the browser via BrowserBase.
    Uses agent abstraction for model predictions.
    Communicates via queues.
    """
    from .simple_browser_env import SimpleBrowserEnv

    global _active_threads
    with _active_threads_lock:
        _active_threads += 1
        logger.warning(
            f"[pod:{_process_id}] Browser worker started. Active threads: {_active_threads}"
        )

    log_memory("browser_worker_start")
    env = None
    try:
        agent = create_agent(settings)
        agent.reset()
        if context:
            agent.set_context(context)

        env = SimpleBrowserEnv(
            browserbase_api_key=browserbase_api_key,
            browserbase_project_id=browserbase_project_id,
            start_url=start_url,
            goal=goal,
            viewport_width=1280,
            viewport_height=720,
        )
        obs, info = env.reset(start_url=start_url, goal=goal)
        print(f"[WORKER] Browser initialized successfully")

        result_queue.put(
            {
                "type": "init_complete",
                "screenshot_base64": _obs_screenshot_to_base64(obs),
                "url": obs["url"],
                "live_view_url": info.get("live_view_url"),
                "bb_session_id": info.get("bb_session_id"),
            }
        )

        # Cache the post-action observation 
        cached_obs = None

        # Main loop - process commands
        while True:
            try:
                command = command_queue.get(timeout=1)
            except:
                continue

            if command["type"] == "stop":
                break

            elif command["type"] == "step":
                try:
                    step_num = command.get("step", 0)
                    print(f"[WORKER] Starting step {step_num}...")

                    # Check if session was paused or stopped before starting this step
                    if session_id:
                        session = session_manager.get_session_light(session_id)
                        if session and session.status in ("paused", "stopped"):
                            result_queue.put(
                                {
                                    "type": "step_result",
                                    "step": step_num,
                                    "cancelled": True,
                                }
                            )
                            continue

                    if cached_obs is not None:
                        obs = cached_obs
                        cached_obs = None
                    else:
                        obs = env.get_obs()
                    screenshot_base64 = _obs_screenshot_to_base64(obs)
                    obs["screenshot_base64"] = screenshot_base64
                    current_url = obs["url"]

                    print(f"[WORKER] Calling agent.predict_action()...")

                    # Call agent
                    raw_response, action_result = agent.predict_action(obs)

                    action_output = action_result.get("action_output")
                    thought = action_result.get("thought", "")
                    action_str = action_result.get("action_str", "")
                    action_description = action_result.get("action_description", "")

                    # Get action name from the action object
                    action_name = (
                        action_output.action.name if action_output else "unknown"
                    )

                    print(
                        f"[WORKER] Agent response - action: {action_name}, thought: {thought[:50]}..."
                    )

                    # Handle click coordinates for annotation
                    click_coords = None
                    annotated_screenshot = screenshot_base64

                    if action_output and isinstance(action_output.action, MouseClick):
                        click_coords = {
                            "x": action_output.action.x,
                            "y": action_output.action.y,
                        }
                        try:
                            annotated_screenshot = add_click_indicator(
                                screenshot_base64,
                                action_output.action.x,
                                action_output.action.y,
                                (720, 1280),
                            )
                        except Exception as e:
                            print(f"Could not add click indicator: {e}")

                    is_final = False
                    final_message = None
                    agent_message = None
                    is_answer = False

                    if action_output:
                        action = action_output.action
                        if isinstance(action, SendMsgToUser):
                            agent_message = action.msg
                            if "[ANSWER]" in action.msg:
                                final_message = action.msg.replace("[ANSWER]", "", 1)
                                agent_message = final_message.strip()
                                is_final = True
                                is_answer = True
                            elif "[EXIT]" in action.msg:
                                final_message = action.msg.replace("[EXIT]", "", 1)
                                is_final = True
                            # is_answer = "[ANSWER]" in action.msg
                            # # Strip "[ANSWER]"
                            # agent_message = agent_message.replace("[ANSWER]", "", 1)

                            # if "[EXIT]" in action.msg:
                            # final_message = action.msg.replace("[EXIT]", "", 1)
                            # is_final = True

                        elif isinstance(action, ReportInfeasible):
                            is_final = True
                            final_message = (
                                f"Cannot complete: {action.infeasibility_reason}"
                            )

                    # Send action preview
                    result_queue.put(
                        {
                            "type": "action_preview",
                            "step": step_num,
                            "thought": thought,
                            "action_str": action_str,
                            "action_name": action_name,
                            "action_description": action_description,
                            "click_coords": click_coords,
                            "screenshot_base64": screenshot_base64,
                        }
                    )

                    # Build result
                    result = {
                        "type": "step_result",
                        "step": step_num,
                        "thought": thought,
                        "action_str": action_str,
                        "action_name": action_name,
                        "action_description": action_description,
                        "screenshot_base64": screenshot_base64,
                        "annotated_screenshot": annotated_screenshot,
                        "click_coords": click_coords,
                        "url": current_url,
                        "is_final": is_final,
                        "final_message": final_message,
                        "agent_message": agent_message,
                        "is_answer": is_answer,
                        "raw_response": raw_response,
                    }

                    # Check if session was paused or stopped after prediction, before executing action
                    if session_id:
                        session = session_manager.get_session_light(session_id)
                        if session and session.status in ("paused", "stopped"):
                            result["cancelled"] = True
                            result_queue.put(result)
                            continue

                    # Execute action if not final
                    if (
                        not is_final
                        and action_output
                        and action_name not in ("unknown", "")
                    ):
                        print(f"[WORKER] Executing action: {action_name}")
                        new_obs, step_info = env.step(action_output.action)
                        if "error" in new_obs:
                            result["execution_success"] = False
                            result["execution_error"] = new_obs.get("error", "Unknown")
                        else:
                            result["execution_success"] = step_info.get(
                                "execution_success", True
                            )
                            if step_info.get("execution_error"):
                                result["execution_error"] = step_info["execution_error"]
                            result["screenshot_base64"] = _obs_screenshot_to_base64(
                                new_obs
                            )
                            result["url"] = new_obs["url"]
                            if step_info.get("live_view_url"):
                                result["live_view_url"] = step_info["live_view_url"]
                            cached_obs = new_obs

                    result_queue.put(result)
                    print(f"[WORKER] Step {step_num} complete")

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    result_queue.put(
                        {
                            "type": "step_result",
                            "step": command.get("step", 0),
                            "error": str(e),
                        }
                    )

            elif command["type"] == "new_task":
                try:
                    cached_obs = None
                    new_goal = command.get("goal", goal)
                    new_context = command.get("context")
                    goal = new_goal

                    print(f"[WORKER] Starting new task: {goal[:50]}...")

                    env.set_goal(goal)
                    agent.reset()
                    if new_context:
                        agent.set_context(new_context)

                    obs = env.get_obs()
                    result_queue.put(
                        {
                            "type": "new_task_ready",
                            "screenshot_base64": _obs_screenshot_to_base64(obs),
                            "url": obs["url"],
                        }
                    )
                    print(f"[WORKER] New task ready, continuing from {obs['url']}")

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    result_queue.put(
                        {
                            "type": "new_task_error",
                            "error": str(e),
                        }
                    )

            elif command["type"] == "navigate":
                cached_obs = None
                reply_queue = command.get("reply_queue")
                try:
                    url = command.get("url", "about:blank")
                    env.page.goto(url, wait_until="domcontentloaded")
                    obs = env.get_obs()
                    result = {
                        "type": "navigate_done",
                        "url": obs["url"],
                        "screenshot_base64": _obs_screenshot_to_base64(obs),
                    }
                    if reply_queue is not None:
                        reply_queue.put(result)
                    else:
                        result_queue.put(result)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    err_result = {"type": "navigate_done", "error": str(e)}
                    if reply_queue is not None:
                        reply_queue.put(err_result)
                    else:
                        result_queue.put(err_result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        result_queue.put(
            {
                "type": "init_error",
                "error": str(e),
            }
        )

    finally:
        if env is not None:
            env.close()
        agent.reset()
        del agent
        log_memory("browser_worker_end")
        with _active_threads_lock:
            _active_threads -= 1
            logger.warning(
                f"[pod:{_process_id}] Browser worker exited. Active threads: {_active_threads}"
            )


class BrowserService:
    """
    Manages a BrowserBase browser session in a separate thread.

    Reference: https://playwright.dev/python/docs/library#threading
    """

    def __init__(
        self,
        browserbase_api_key: str,
        browserbase_project_id: str,
        start_url: str = "https://www.google.com",
        goal: str = "",
        context: dict[str, Any] | None = None,
        session_id: Optional[str] = None,
    ):
        self.browserbase_api_key = browserbase_api_key
        self.browserbase_project_id = browserbase_project_id
        self.start_url = start_url
        self.goal = goal
        self.context = context
        self.session_id = session_id

        self.command_queue: Optional[queue.Queue] = None
        self.result_queue: Optional[queue.Queue] = None
        self.thread: Optional[Thread] = None
        self.step_count = 0
        self.is_running = False

    def initialize(self) -> dict[str, Any]:
        """Start the browser thread and wait for initialization."""
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.thread = Thread(
            target=browser_worker,
            args=(
                self.command_queue,
                self.result_queue,
                self.browserbase_api_key,
                self.browserbase_project_id,
                self.start_url,
                self.goal,
                self.context,
                self.session_id,
            ),
            daemon=True,
        )
        self.thread.start()

        # Wait for initialization result (blocking)
        while True:
            try:
                result = self.result_queue.get(timeout=120)

                if result["type"] == "init_complete":
                    self.is_running = True
                    return result
                elif result["type"] == "init_error":
                    raise Exception(result["error"])

            except queue.Empty:
                if self.thread and not self.thread.is_alive():
                    raise Exception("Browser thread died during initialization")
                raise

    def predict_and_execute(
        self, on_action_preview: Optional[Callable] = None
    ) -> dict[str, Any]:
        """
        Send a step command and wait for result.

        Args:
            on_action_preview: Optional callback for action preview (click coords before execution)
        """
        if not self.is_running:
            return {"error": "Browser not running"}

        if self.thread and not self.thread.is_alive():
            return {"error": "Browser thread has died"}

        self.step_count += 1
        self.command_queue.put({"type": "step", "step": self.step_count})

        # Wait for results
        try:
            while True:
                result = self.result_queue.get(
                    timeout=360
                )  # 6 min for Modal cold starts

                if result.get("type") == "action_preview":
                    if on_action_preview:
                        on_action_preview(result)
                    continue
                elif result.get("type") == "step_result":
                    return result
                else:
                    return result

        except queue.Empty as e:
            if self.thread and not self.thread.is_alive():
                return {"error": f"Browser thread died: {e}"}
            return {"error": f"Timeout waiting for browser response: {e}"}

    def get_current_screenshot(self) -> Optional[str]:
        """Not available with thread-based approach."""
        return None

    def start_new_task(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Start a new task on the existing browser session.

        Used for follow-up queries to continue from current browser state
        instead of creating a new browser session.
        """
        if not self.is_running:
            return {"error": "Browser not running"}

        if self.thread and not self.thread.is_alive():
            return {"error": "Browser thread has died"}

        self.goal = goal
        self.context = context
        self.step_count = 0

        self.command_queue.put(
            {
                "type": "new_task",
                "goal": goal,
                "context": context,
            }
        )

        # Wait for ready signal
        try:
            result = self.result_queue.get(timeout=30)
            if result.get("type") == "new_task_error":
                return {"error": result.get("error", "Unknown error")}
            return result
        except queue.Empty:
            return {"error": "Timeout waiting for new task ready signal"}

    def navigate(self, url: str = "about:blank") -> Optional[dict[str, Any]]:
        """
        Navigate the browser to a URL (e.g. about:blank) without closing the session.
        Uses a reply_queue so the result is not consumed by predict_and_execute.
        """
        if not self.is_running:
            return {"error": "Browser not running"}

        if self.thread and not self.thread.is_alive():
            return {"error": "Browser thread has died"}

        reply_queue = queue.Queue()
        self.command_queue.put(
            {
                "type": "navigate",
                "url": url,
                "reply_queue": reply_queue,
            }
        )

        try:
            result = reply_queue.get(timeout=30)
            if result.get("error"):
                return {"error": result.get("error")}
            return result
        except queue.Empty:
            return {"error": "Timeout waiting for navigate"}

    def close(self):
        """Stop the browser thread."""
        self.is_running = False

        if self.command_queue:
            try:
                self.command_queue.put({"type": "stop"})
            except:
                pass

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
