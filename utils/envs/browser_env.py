"""
Browser environments for web agent evaluation.

Two concrete implementations:
  - BrowserbaseEnv: connects to Browserbase (cloud, stealth proxies, CAPTCHA solving)
  - LocalBrowserEnv: launches a local Chromium (headless or headed, no proxies)

Both share the same interface:
  env = BrowserbaseEnv(start_url=..., goal=...)  # or LocalBrowserEnv(...)
  obs, info = env.reset()
  obs = env.step(action)
  env.close()

Observations include screenshot, axtree, and extra_element_properties when
extract_axtree=True (default). Set extract_axtree=False for visual-only agents.
"""
import asyncio
import base64
import logging
import os
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from agent.actions import ALL_ACTIONS, BrowserNav, MouseClick, SendMsgToUser
from utils.envs.action_executor import execute_action
from utils.axtree import extract_axtree, extract_screenshot, MarkingError, EXTRACT_OBS_MAX_TRIES

logger = logging.getLogger(__name__)


def _start_playwright():
    asyncio._set_running_loop(None)
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return sync_playwright().start()


def _wait_ready(page, timeout_ms: int = 10000):
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
        return
    except PlaywrightTimeoutError:
        pass
    try:
        page.wait_for_load_state("load", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        pass


def _take_screenshot(page) -> np.ndarray:
    try:
        cdp = page.context.new_cdp_session(page)
        result = cdp.send("Page.captureScreenshot", {"format": "png"})
        cdp.detach()
        raw = base64.b64decode(result["data"])
    except Exception:
        raw = page.screenshot(timeout=10000, animations="disabled")
    return np.array(Image.open(BytesIO(raw)).convert("RGB"))


class BrowserEnv(ABC):
    """Base browser environment."""

    def __init__(
        self,
        start_url: str = "about:blank",
        goal: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        extract_axtree: bool = False,
        robust_navigation: bool = False,
    ):
        self.start_url = start_url
        self.goal = goal
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.extract_axtree = extract_axtree
        self.robust_navigation = robust_navigation

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.last_action_error = ""
        self.step_count = 0

    @abstractmethod
    def _launch(self):
        """Launch browser and set self.playwright, self.browser, self.context, self.page."""

    def reset(self, start_url: str | None = None, goal: str | None = None) -> tuple[dict, dict]:
        if start_url is not None:
            self.start_url = start_url
        if goal is not None:
            self.goal = goal

        self.close()
        self._launch()

        self.page.set_viewport_size({"width": self.viewport_width, "height": self.viewport_height})
        self.page.set_default_timeout(120000)

        self.goal = self._navigate_to_start(self.start_url, self.goal)

        _wait_ready(self.page)
        self.last_action_error = ""
        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _navigate_to_start(self, url: str, goal: str) -> str:
        """Navigate to start_url with fallbacks. Returns (possibly modified) goal."""
        if not self.robust_navigation:
            self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            return goal

        # Attempt 1: direct goto
        try:
            self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            return goal
        except Exception as e:
            logger.warning(f"goto failed for {url}: {e}")

        # Attempt 2: warm up via bing.com then retry. Establishing a real
        # HTTP/2 connection first helps with sites that reject cold connections
        # from automated browsers.
        try:
            logger.info(f"Warming up via bing.com before retrying {url}")
            self.page.goto("https://www.bing.com/", timeout=30000, wait_until="domcontentloaded")
            time.sleep(2)
            self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
            return goal
        except Exception as e:
            logger.warning(f"Bing warmup + goto failed for {url}: {e}")

        # Both attempts failed -- let the agent navigate there itself
        logger.warning(f"All navigation attempts failed for {url}. Agent will navigate manually.")
        return f"First, navigate to {url}\n\n{goal}"

    def step(self, action: ALL_ACTIONS) -> dict:
        self.step_count += 1

        new_page = self._execute_with_tab_detection(action)
        if new_page:
            self.page = new_page

        # 1. Wait for JS events / callbacks to fire
        time.sleep(0.5)
        # 2. Wait for domcontentloaded on ALL open pages and frames
        for p in self.context.pages:
            try:
                p.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass
            for frame in p.frames:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
        # 3. Final domcontentloaded on active page + extra buffer
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        time.sleep(0.5)

        return self._get_obs()

    def _execute_with_tab_detection(self, action: ALL_ACTIONS):
        """Execute action, detecting if it opens a new tab."""
        might_open_tab = isinstance(action, MouseClick) or (
            isinstance(action, BrowserNav) and action.nav_type == "new_tab"
        )

        new_page = None
        if might_open_tab:
            try:
                with self.context.expect_page(timeout=2000) as new_page_info:
                    success, error = execute_action(self.page, action)
                new_page = new_page_info.value
                try:
                    new_page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                new_page.bring_to_front()
            except PlaywrightTimeoutError:
                new_page = None
        else:
            success, error = execute_action(self.page, action)

            if isinstance(action, BrowserNav) and action.nav_type == "tab_focus":
                pages = self.context.pages
                if 0 <= action.index < len(pages):
                    new_page = pages[action.index]

        self.last_action_error = error if not success else ""
        return new_page

    def _get_obs(self) -> dict[str, Any]:
        screenshot = _take_screenshot(self.page)

        obs = {
            "screenshot": screenshot,
            "url": self.page.url,
            "goal": self.goal,
            "open_pages_titles": [],
            "open_pages_urls": [],
            "active_page_index": [0],
            "last_action_error": self.last_action_error,
            "axtree_object": {},
            "extra_element_properties": {},
        }

        for i, p in enumerate(self.context.pages):
            try:
                obs["open_pages_titles"].append(p.title())
                obs["open_pages_urls"].append(p.url)
                if p == self.page:
                    obs["active_page_index"] = [i]
            except Exception:
                obs["open_pages_titles"].append("Unknown")
                obs["open_pages_urls"].append("")

        if self.extract_axtree:
            for retries in reversed(range(EXTRACT_OBS_MAX_TRIES)):
                try:
                    axtree, extra = extract_axtree(self.page, lenient=(retries == 0))
                    obs["axtree_object"] = axtree
                    obs["extra_element_properties"] = extra
                    break
                except (MarkingError, Exception) as e:
                    if retries > 0:
                        logger.debug(f"AXTree extraction retry ({retries} left): {e}")
                        time.sleep(0.5)
                    else:
                        logger.warning(f"AXTree extraction failed after all retries: {e}")

        return obs

    @abstractmethod
    def _get_info(self) -> dict[str, Any]:
        """Return env-specific info dict."""

    def close(self):
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
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None


class BrowserbaseEnv(BrowserEnv):
    """Browser environment using Browserbase (cloud, stealth, CAPTCHA solving)."""

    def __init__(
        self,
        start_url: str = "about:blank",
        goal: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        extract_axtree: bool = False,
        api_key: str | None = None,
        project_id: str | None = None,
        native_polyfill: bool = False,
        robust_navigation: bool = False,
    ):
        super().__init__(start_url, goal, viewport_width, viewport_height, extract_axtree, robust_navigation)
        self.api_key = api_key or os.getenv("BROWSERBASE_API_KEY")
        self.project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")
        self.native_polyfill = native_polyfill
        self.bb = None
        self.bb_session = None

    def _launch(self):
        from browserbase import Browserbase

        if not self.api_key or not self.project_id:
            raise ValueError("BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID required")

        self.bb = Browserbase(api_key=self.api_key)
        browser_settings = {"advanced_stealth": True}
        if self.native_polyfill:
            browser_settings["enableNativeSelectPolyfill"] = False
        self.bb_session = self.bb.sessions.create(
            project_id=self.project_id,
            proxies=True,
            browser_settings=browser_settings,
        )
        logger.info(f"BB session: {self.bb_session.id}")

        self.playwright = _start_playwright()
        cdp_url = f"wss://connect.browserbase.com?sessionId={self.bb_session.id}&apiKey={self.api_key}"
        self.browser = self.playwright.chromium.connect_over_cdp(cdp_url)

        if self.browser.contexts:
            self.context = self.browser.contexts[0]
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        else:
            self.context = self.browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height}
            )
            self.page = self.context.new_page()

    def _get_info(self) -> dict[str, Any]:
        info = {"bb_session_id": self.bb_session.id if self.bb_session else None}
        if self.bb and self.bb_session:
            try:
                debug = self.bb.sessions.debug(self.bb_session.id)
                if hasattr(debug, "pages") and debug.pages:
                    info["live_view_url"] = debug.pages[0].debugger_fullscreen_url
            except Exception:
                pass
        return info

    def close(self):
        if self.bb and self.bb_session:
            try:
                self.bb.sessions.update(self.bb_session.id, status="REQUEST_RELEASE")
            except Exception:
                pass
        super().close()
        self.bb = None
        self.bb_session = None


class SimpleEnv(BrowserEnv):
    """Browser environment using a local Chromium instance."""

    STEALTH_ARGS = ["--disable-blink-features=AutomationControlled"]

    def __init__(
        self,
        start_url: str = "about:blank",
        goal: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        extract_axtree: bool = False,
        headless: bool = True,
        channel: str | None = None,
    ):
        super().__init__(start_url, goal, viewport_width, viewport_height, extract_axtree)
        self.headless = headless
        self.channel = channel

    def _launch(self):
        self.playwright = _start_playwright()
        launch_opts: dict = {
            "headless": self.headless,
            "args": self.STEALTH_ARGS,
        }
        if self.channel:
            launch_opts["channel"] = self.channel
        self.browser = self.playwright.chromium.launch(**launch_opts)
        self.context = self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self.page = self.context.new_page()

    def _get_info(self) -> dict[str, Any]:
        return {}