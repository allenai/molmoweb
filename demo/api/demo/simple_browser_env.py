"""Standalone browser environment for agent-driven browser automation."""

import base64
import os
import time
from io import BytesIO
from typing import Any, Optional

import numpy as np
from agent.actions import ActionOutput, BrowserNav, MouseClick
from browserbase import Browserbase
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .browser_service import (
    execute_action_from_output,
    start_playwright_sync,
    take_screenshot_fast,
)
from .mem_util import log_memory


def _wait_for_page_ready(page, timeout_ms: int = 10000) -> None:
    """Best-effort wait for page load to settle before taking screenshots."""
    if not page or page.is_closed():
        return

    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    try:
        page.wait_for_load_state("load", timeout=timeout_ms)
    except Exception:
        pass

    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 3000))
    except Exception:
        pass


class SimpleBrowserEnv:
    """Standalone environment for agent-driven browser control.

    One browser session: reset() -> get_obs() / step(action) -> close().
    Demo worker runs this in a single thread; no cross-thread access.
    """

    def __init__(
        self,
        browserbase_api_key: str,
        browserbase_project_id: str,
        start_url: str = "about:blank",
        goal: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        self.browserbase_api_key = browserbase_api_key
        self.browserbase_project_id = browserbase_project_id
        self.start_url = start_url
        self.goal = goal
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self.playwright = None
        self.browser = None
        self.browser_context = None
        self.page = None
        self.bb = None
        self.bb_session = None
        self.last_action_error = ""
        self.is_running = False

    def reset(
        self,
        start_url: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a fresh browser session and navigate to the start URL."""
        if start_url is not None:
            self.start_url = start_url
        if goal is not None:
            self.goal = goal

        self.close()

        os.environ["BROWSERBASE_API_KEY"] = self.browserbase_api_key
        os.environ["BROWSERBASE_PROJECT_ID"] = self.browserbase_project_id

        self.bb = Browserbase(api_key=self.browserbase_api_key)
        self.bb_session = self.bb.sessions.create(
            project_id=self.browserbase_project_id,
            proxies=True,
            browser_settings={"advanced_stealth": True},
        )

        self.playwright = start_playwright_sync()
        cdp_url = (
            f"wss://connect.browserbase.com"
            f"?sessionId={self.bb_session.id}&apiKey={self.browserbase_api_key}"
        )
        self.browser = self.playwright.chromium.connect_over_cdp(cdp_url)

        if self.browser.contexts:
            self.browser_context = self.browser.contexts[0]
            self.page = (
                self.browser_context.pages[0]
                if self.browser_context.pages
                else self.browser_context.new_page()
            )
        else:
            self.browser_context = self.browser.new_context(
                viewport={
                    "width": self.viewport_width,
                    "height": self.viewport_height,
                }
            )
            self.page = self.browser_context.new_page()

        self.page.set_viewport_size(
            {"width": self.viewport_width, "height": self.viewport_height}
        )
        self.page.set_default_timeout(120000)
        self.page.goto(self.start_url, wait_until="domcontentloaded")
        _wait_for_page_ready(self.page, timeout_ms=5000)
        # time.sleep(1)
        screenshot_bytes = take_screenshot_fast(self.page)
        obs = self._get_obs(screenshot_bytes)

        live_view_url = None
        try:
            debug_info = self.bb.sessions.debug(self.bb_session.id)
            if hasattr(debug_info, "pages") and debug_info.pages:
                live_view_url = debug_info.pages[0].debugger_fullscreen_url
            elif hasattr(debug_info, "debugger_fullscreen_url"):
                live_view_url = debug_info.debugger_fullscreen_url
        except Exception as e:
            print(f"[BrowserEnv] Could not get live view URL: {e}")

        self.last_action_error = ""
        self.is_running = True

        info = {
            "live_view_url": live_view_url,
            "bb_session_id": self.bb_session.id,
        }
        return obs, info

    def _get_obs(
        self, screenshot_bytes: Optional[bytes] = None
    ) -> dict[str, Any]:
        """Build observation dict from the current page state (same shape as build_observation)."""
        if screenshot_bytes is None:
            screenshot_bytes = take_screenshot_fast(self.page)
        img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
        screenshot_np = np.array(img)

        current_url = self.page.url
        open_pages_titles = []
        open_pages_urls = []
        active_page_index = 0

        for i, p in enumerate(self.page.context.pages):
            try:
                open_pages_titles.append(p.title())
                open_pages_urls.append(p.url)
                if p == self.page:
                    active_page_index = i
            except Exception:
                open_pages_titles.append("Unknown")
                open_pages_urls.append("")

        return {
            "goal": self.goal,
            "screenshot": screenshot_np,
            "url": current_url,
            "open_pages_titles": open_pages_titles,
            "open_pages_urls": open_pages_urls,
            "active_page_index": [active_page_index],
            "last_action_error": self.last_action_error,
        }

    def get_obs(self) -> dict[str, Any]:
        """Current observation without executing an action."""
        return self._get_obs(None)

    def set_goal(self, goal: str) -> None:
        """Update goal (e.g. for new_task without closing browser)."""
        self.goal = goal

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one action; return (obs, info). New-tab handling, no double-click."""
        if not self.is_running or not self.page:
            return ({"error": "Browser not running"}, {})

        action_output = ActionOutput(action=action, thought="")
        action_name = action.name if hasattr(action, "name") else "unknown"

        if action_name in ("unknown", ""):
            obs = self.get_obs()
            return obs, {}

        time.sleep(0.1)
        new_page_index = None
        success, error = True, ""

        is_click_action = isinstance(action, MouseClick)
        if is_click_action:
            try:
                with self.browser_context.expect_page(
                    timeout=2000
                ) as new_page_info:
                    success, error = execute_action_from_output(
                        self.page, action_output
                    )
                new_page = new_page_info.value
                try:
                    # new_page.wait_for_load_state(
                    #     "domcontentloaded", timeout=5000
                    # )
                    _wait_for_page_ready(new_page, timeout_ms=5000)
                except Exception as e:
                    print(f"[BrowserEnv] Timeout waiting for new page: {e}")
                new_page.bring_to_front()
                self.page = new_page
                new_page_index = len(self.browser_context.pages) - 1
            except PlaywrightTimeoutError:
                # Click already ran once; do not execute again.
                new_page = None
        else:
            success, error = execute_action_from_output(
                self.page, action_output
            )
            if (
                isinstance(action, BrowserNav)
                and action.nav_type == "tab_focus"
            ):
                pages = self.browser_context.pages
                if 0 <= action.index < len(pages):
                    self.page = pages[action.index]
                    new_page_index = action.index

        if error:
            self.last_action_error = error
        else:
            self.last_action_error = ""

        live_view_url = None
        if new_page_index is not None and self.bb and self.bb_session:
            try:
                debug_info = self.bb.sessions.debug(self.bb_session.id)
                if (
                    hasattr(debug_info, "pages")
                    and debug_info.pages
                    and new_page_index < len(debug_info.pages)
                ):
                    live_view_url = debug_info.pages[
                        new_page_index
                    ].debugger_fullscreen_url
            except Exception as e:
                print(
                    f"[BrowserEnv] Could not get live view URL for new tab: {e}"
                )

        # time.sleep(1)
        _wait_for_page_ready(self.page, timeout_ms=5000)
        post_screenshot_bytes = take_screenshot_fast(self.page)
        obs = self._get_obs(post_screenshot_bytes)

        info = {
            "live_view_url": live_view_url,
            "execution_success": success,
            "execution_error": error,
        }
        return obs, info

    def close(self) -> None:
        """Tear down the browser session."""
        log_memory("browser_env_closing")
        self.is_running = False

        try:
            if self.browser:
                self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass
        if self.bb and self.bb_session:
            try:
                self.bb.sessions.update(
                    self.bb_session.id, status="REQUEST_RELEASE"
                )
            except Exception:
                pass

        self.playwright = None
        self.browser = None
        self.browser_context = None
        self.page = None
        self.bb = None
        self.bb_session = None
