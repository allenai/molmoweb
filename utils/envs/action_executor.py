"""
Action execution for browser environments.

All coordinate-based actions store pixel values. The executor just runs them.
Each agent is responsible for converting its own coordinate system to pixels.
"""
import time

from agent.actions import (
    ALL_ACTIONS,
    BrowserNav,
    Click,
    GeminiTypeTextAt,
    Goto,
    HoverAt,
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


def execute_action(page, action: ALL_ACTIONS) -> tuple[bool, str]:
    """Execute an action on a Playwright page. Returns (success, error_msg).

    All coordinate-based actions must already contain pixel values.
    """
    try:
        if isinstance(action, Click):
            return _click_by_bid(page, action)

        if isinstance(action, MouseClick):
            if action.click_type == "double":
                page.mouse.dblclick(action.x, action.y)
            else:
                page.mouse.click(action.x, action.y)
            time.sleep(1)
            return True, ""

        if isinstance(action, MouseMove):
            page.mouse.move(action.x, action.y)
            return True, ""

        if isinstance(action, HoverAt):
            page.mouse.move(action.x, action.y)
            time.sleep(action.duration)
            return True, ""

        if isinstance(action, MouseDragAndDrop):
            page.mouse.move(action.from_x, action.from_y)
            page.mouse.down()
            page.mouse.move(action.to_x, action.to_y)
            page.mouse.up()
            return True, ""

        if isinstance(action, Scroll):
            page.mouse.wheel(action.delta_x, action.delta_y)
            return True, ""

        if isinstance(action, ScrollAt):
            page.mouse.move(action.x, action.y)
            page.mouse.wheel(action.delta_x, action.delta_y)
            return True, ""

        if isinstance(action, GeminiTypeTextAt):
            return _type_at(page, action)

        if isinstance(action, KeyboardType):
            page.keyboard.type(action.text)
            return True, ""

        if isinstance(action, KeyboardPress):
            page.keyboard.press(action.key)
            if action.key.lower() == "enter":
                time.sleep(1)
            return True, ""

        if isinstance(action, Goto):
            page.goto(action.url, wait_until="domcontentloaded")
            return True, ""

        if isinstance(action, BrowserNav):
            if action.nav_type == "go_back":
                page.go_back()
            elif action.nav_type == "new_tab":
                page.context.new_page()
            elif action.nav_type == "tab_focus":
                pages = page.context.pages
                if 0 <= action.index < len(pages):
                    pages[action.index].bring_to_front()
            return True, ""

        if isinstance(action, Noop):
            time.sleep(5)
            return True, ""

        if isinstance(action, (SendMsgToUser, ReportInfeasible)):
            return True, ""

        return False, f"Unknown action type: {type(action)}"

    except Exception as e:
        return False, str(e)


def _click_by_bid(page, action: Click) -> tuple[bool, str]:
    try:
        elem = page.locator(f'[bid="{action.bid}"]').first
        if action.click_type == "double":
            elem.dblclick(button=action.button, timeout=2000)
        else:
            elem.click(button=action.button, timeout=2000)
        time.sleep(1)
        return True, ""
    except Exception:
        try:
            elem = page.locator(f'[bid="{action.bid}"]').first
            if action.click_type == "double":
                elem.dblclick(button=action.button, force=True, timeout=2000)
            else:
                elem.click(button=action.button, force=True, timeout=2000)
            time.sleep(1)
            return True, ""
        except Exception as e2:
            return False, str(e2)


def _type_at(page, action: GeminiTypeTextAt) -> tuple[bool, str]:
    try:
        page.mouse.click(action.x, action.y)
        time.sleep(0.3)

        if action.clear_before_typing:
            page.keyboard.press("ControlOrMeta+a")
            time.sleep(0.05)
            page.keyboard.press("Backspace")
            time.sleep(0.05)

        if action.text:
            page.keyboard.type(action.text)
            time.sleep(0.1)

        if action.press_enter:
            page.keyboard.press("Enter")

        return True, ""
    except Exception as e:
        return False, str(e)
