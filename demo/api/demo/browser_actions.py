"""Demo-specific action execution: delegates to core executor with safety checks."""

import re
import time

from agent.actions import ALL_ACTIONS, GeminiTypeTextAt, KeyboardPress, KeyboardType, Noop
from utils.envs.action_executor import execute_action

_CC_PATTERN = r"credit.?card|card.?number|cc.?num|cvv|cvc|ccv|security.?code"


def _check_sensitive_field(page) -> str | None:
    """Return an error message if the focused element is sensitive, else None."""
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


def execute_action_for_demo(page, action: ALL_ACTIONS) -> tuple[bool, str]:
    """
    Run an action like utils.envs.action_executor.execute_action, with demo guards:
    refuse typing into password / credit-card fields; noop waits for load.
    """
    if isinstance(action, Noop):
        try:
            page.wait_for_load_state("load", timeout=15_000)
        except Exception:
            pass
        return True, ""

    if isinstance(action, (KeyboardType, KeyboardPress)):
        err = _check_sensitive_field(page)
        if err:
            return False, err
        return execute_action(page, action)

    if isinstance(action, GeminiTypeTextAt):
        try:
            page.mouse.click(action.x, action.y)
            time.sleep(0.3)
            err = _check_sensitive_field(page)
            if err:
                return False, err
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

    return execute_action(page, action)
