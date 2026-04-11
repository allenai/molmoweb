"""Browserbase-backed env for the interactive demo; extends core BrowserbaseEnv."""

from typing import Any

from agent.actions import ALL_ACTIONS, BrowserNav, MouseClick
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from utils.envs.browser_env import BrowserbaseEnv

from .browser_actions import execute_action_for_demo


class DemoBrowserbaseEnv(BrowserbaseEnv):
    """
    Same as BrowserbaseEnv (CAPTCHA handling, stealth) with a demo-friendly API:
    ``reset`` / ``step`` return extra info, ``get_obs`` / ``set_goal``, and
    stricter typing safety via execute_action_for_demo.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        start_url: str = "about:blank",
        goal: str = "",
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        super().__init__(
            start_url=start_url,
            goal=goal,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            extract_axtree=False,
            api_key=api_key,
            project_id=project_id,
        )
        self.is_running = False

    def _execute_with_tab_detection(self, action: ALL_ACTIONS):
        might_open_tab = isinstance(action, MouseClick) or (
            isinstance(action, BrowserNav) and action.nav_type == "new_tab"
        )

        new_page = None
        if might_open_tab:
            try:
                with self.context.expect_page(timeout=2000) as new_page_info:
                    success, error = execute_action_for_demo(self.page, action)
                new_page = new_page_info.value
                try:
                    new_page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                new_page.bring_to_front()
            except PlaywrightTimeoutError:
                new_page = None
        else:
            success, error = execute_action_for_demo(self.page, action)

            if isinstance(action, BrowserNav) and action.nav_type == "tab_focus":
                pages = self.context.pages
                if 0 <= action.index < len(pages):
                    new_page = pages[action.index]

        self.last_action_error = error if not success else ""
        return new_page

    def reset(
        self,
        start_url: str | None = None,
        goal: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(start_url=start_url, goal=goal)
        self.is_running = True
        return obs, info

    def step(self, action: ALL_ACTIONS) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.is_running or not self.page:
            return {"error": "Browser not running"}, {}

        n_pages_before = len(self.context.pages)
        page_before = self.page
        obs = super().step(action)
        live_view_url = None
        if len(self.context.pages) > n_pages_before or self.page is not page_before:
            live_view_url = self._live_view_for_active_tab(obs)

        err = obs.get("last_action_error") or ""
        info: dict[str, Any] = {
            "live_view_url": live_view_url,
            "execution_success": not bool(err),
            "execution_error": err,
        }
        return obs, info

    def _live_view_for_active_tab(self, obs: dict[str, Any]) -> str | None:
        if not self.bb or not self.bb_session:
            return None
        try:
            idx = int(obs["active_page_index"][0])
            debug_info = self.bb.sessions.debug(self.bb_session.id)
            if hasattr(debug_info, "pages") and debug_info.pages and idx < len(debug_info.pages):
                return debug_info.pages[idx].debugger_fullscreen_url
        except Exception:
            pass
        return None

    def get_obs(self) -> dict[str, Any]:
        if not self.page:
            return {"error": "Browser not running"}
        return self._get_obs()

    def set_goal(self, goal: str) -> None:
        self.goal = goal

    def close(self) -> None:
        self.is_running = False
        super().close()
