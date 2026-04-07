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
from datetime import UTC, datetime, timezone
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
logger.setLevel(logging.INFO)


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

        # Always use the Browserbase-provisioned default context and page.
        # Creating a new context bypasses Browserbase's built-in fingerprinting
        # and stealth configuration. contexts[0]/pages[0] are guaranteed to
        # exist after connect_over_cdp with a live Browserbase session.
        self.context = self.browser.contexts[0]
        self.page = self.context.pages[0]

        # listen for captcha solving events
        self.captcha_timeout_seconds = 30
        self.cur_captcha_events = []

        # Attach CAPTCHA listener to the initial page and auto-attach to all
        # future pages (new tabs, popups) so we never miss a CAPTCHA event.
        self._attach_captcha_listener(self.page)
        self.context.on("page", lambda p: self._attach_captcha_listener(p))

    def step(self, action: ALL_ACTIONS) -> dict:
        self.step_count += 1

        new_page = self._execute_with_tab_detection(action)
        if new_page:
            self.page = new_page

        # Wait for CAPTCHA solving to complete BEFORE any CDP/load-state
        # operations. The CAPTCHA is triggered by the action above, so we must
        # check after execution. Anti-bot systems like Kasada detect continued
        # automation (load-state polls, DOM queries) during the challenge.
        self._wait_for_captcha_if_needed()

        # Now safe to wait for page load
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
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        time.sleep(0.5)

        return self._get_obs()

    def _attach_captcha_listener(self, page):
        """Attach a console listener to detect Browserbase CAPTCHA solving events."""

        def handle_console(msg):
            if msg.text == "browserbase-solving-started":
                print("🧩 CAPTCHA solving started")
                self.cur_captcha_events.append(
                    {
                        "event": "solving-started",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            elif msg.text == "browserbase-solving-finished":
                print("🔓 CAPTCHA solving finished")
                self.cur_captcha_events.append(
                    {
                        "event": "solving-finished",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        try:
            page.on("console", handle_console)
        except Exception as e:
            print(f"❌ Could not attach CAPTCHA listener to page: {e}")

    def _wait_for_captcha_if_needed(self):
        """
        Check for and wait on CAPTCHA solving BEFORE any other page operations.

        Called immediately after action execution. Pauses ALL Playwright/CDP
        traffic while Browserbase's solver works, so anti-bot systems like
        Kasada don't detect continued automation during the challenge.

        Browserbase may fire many rapid solving-started/solving-finished pairs
        (multiple challenges, retries). We can't just check for ANY "finished"
        event — we must wait for the entire captcha storm to SETTLE (no new
        events for CAPTCHA_SETTLE_SECONDS) before resuming automation.
        """
        CAPTCHA_SETTLE_SECONDS = 3.0

        # Pump the event loop to flush any pending console events
        try:
            self.context.cookies()
        except Exception:
            pass

        # Grace period: give Browserbase time to detect the CAPTCHA and fire
        # the console event. Without this, the event may arrive after we've
        # already moved on, causing us to send CDP commands that disrupt the
        # solver. We check every 0.5s and break early if an event arrives.
        if not self.cur_captcha_events:
            for _ in range(3):
                time.sleep(0.5)
                try:
                    self.context.cookies()
                except Exception:
                    pass
                if self.cur_captcha_events:
                    break

        # No captcha activity detected — continue normally
        if not self.cur_captcha_events:
            return

        # Captcha activity detected — wait for it to FULLY settle.
        # We track the event count and wait until no new events have arrived
        # for CAPTCHA_SETTLE_SECONDS, AND the last event is "solving-finished".
        # This prevents resuming in the middle of rapid-fire captcha retries
        # where started/finished pairs accumulate but new challenges keep coming.
        print("⏳ CAPTCHA activity detected — pausing automation until fully settled...")
        start_time = time.time()
        last_event_count = len(self.cur_captcha_events)
        last_change_time = time.time()

        while time.time() - start_time < self.captcha_timeout_seconds:
            time.sleep(0.5)
            try:
                self.context.cookies()
            except Exception:
                pass

            current_count = len(self.cur_captcha_events)
            if current_count != last_event_count:
                # New events arrived — reset the settling timer
                last_event_count = current_count
                last_change_time = time.time()
                continue

            # Check if settled: no new events for CAPTCHA_SETTLE_SECONDS
            time_since_last = time.time() - last_change_time
            if time_since_last >= CAPTCHA_SETTLE_SECONDS:
                last_event = (
                    self.cur_captcha_events[-1].get("event", "")
                    if self.cur_captcha_events
                    else ""
                )
                if last_event == "solving-finished":
                    elapsed = time.time() - start_time
                    print(
                        f"✅ CAPTCHA settled in {elapsed:.1f}s "
                        f"({last_event_count} events) — resuming automation"
                    )
                    break
                # Last event is solving-started — solver still working, keep waiting
        else:
            print(
                f"🔒 Timeout: CAPTCHA not resolved within {self.captcha_timeout_seconds}s "
                f"({len(self.cur_captcha_events)} events seen)"
            )

        # Post-captcha: wait for redirect/page load to fully settle
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        time.sleep(1.0)  # buffer for post-captcha navigation

        self.cur_captcha_events = []

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