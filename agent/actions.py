import re
from typing import Any, ClassVar, Literal, Union

from pydantic import BaseModel, Field


NOOP_WAIT_MS = 5000


class Click(BaseModel):
    bid: str = Field(description="The id of the html element to click on.")
    button: Literal["left", "right", "middle"] = Field(
        description="The mouse button to click with."
    )
    click_type: Literal["single", "double"] = Field(
        description="The type of click to perform."
    )

    @property
    def name(self):
        return "click" if self.click_type == "single" else "dblclick"

    def __str__(self) -> str:
        return f"{self.name}(bid={self.bid!r}, button={self.button!r})"

    def describe(self, axtree: dict | None = None, extra_element_properties: dict | None = None) -> str:
        bid_str = f"element with bid={self.bid}"
        if axtree:
            node = _get_node_from_bid(self.bid, axtree)
            if node:
                bid_str = _node2str(node)
        clickable = False
        if extra_element_properties and self.bid in extra_element_properties:
            clickable = extra_element_properties[self.bid].get("clickable", False)
        suffix = "clickable" if clickable else "not clickable"
        return f"{self.button} {self.name} on {bid_str} ({suffix})"


class MouseClick(BaseModel):
    x: float = Field(..., description="The x coordinate in pixels on the viewport.")
    y: float = Field(..., description="The y coordinate in pixels on the viewport.")
    button: Literal["left", "right", "middle"] = Field(
        default="left", description="The mouse button to click with."
    )
    click_type: Literal["single", "double"] = Field(
        default="single", description="The type of click to perform."
    )

    @property
    def name(self) -> str:
        return "mouse_click" if self.click_type == "single" else "mouse_dblclick"

    def __str__(self) -> str:
        return f"{self.name}(x={self.x!r}, y={self.y!r}, button={self.button!r})"

    def describe(self, **_) -> str:
        return f"{self.button} click at (x={self.x}, y={self.y})"


class MouseMove(BaseModel):
    name: ClassVar[str] = "mouse_move"
    x: float = Field(..., description="The x coordinate in pixels.")
    y: float = Field(..., description="The y coordinate in pixels.")

    def __str__(self) -> str:
        return f"{self.name}(x={self.x!r}, y={self.y!r})"

    def describe(self, **_) -> str:
        return f"move mouse to (x={self.x}, y={self.y})"


class HoverAt(BaseModel):
    name: ClassVar[str] = "hover_at"
    x: float = Field(..., description="The x coordinate in pixels.")
    y: float = Field(..., description="The y coordinate in pixels.")
    duration: float = Field(default=1.0, description="Duration in seconds to hover.")

    def __str__(self) -> str:
        return f"{self.name}(x={self.x!r}, y={self.y!r}, duration={self.duration!r})"

    def describe(self, **_) -> str:
        return f"hover at (x={self.x}, y={self.y}) for {self.duration}s"


class Scroll(BaseModel):
    name: ClassVar[str] = "scroll"
    delta_x: float = Field(description="Pixels to scroll horizontally. Positive=right, negative=left.")
    delta_y: float = Field(description="Pixels to scroll vertically. Positive=down, negative=up.")

    def __str__(self) -> str:
        return f"{self.name}(delta_x={self.delta_x!r}, delta_y={self.delta_y!r})"

    def describe(self, **_) -> str:
        if self.delta_x == 0:
            d = "down" if self.delta_y > 0 else "up"
            return f"scroll {d} by {abs(self.delta_y)} pixels"
        elif self.delta_y == 0:
            d = "right" if self.delta_x > 0 else "left"
            return f"scroll {d} by {abs(self.delta_x)} pixels"
        return f"scroll ({self.delta_x}, {self.delta_y}) pixels"


class ScrollAt(BaseModel):
    name: ClassVar[str] = "scroll_at"
    x: float = Field(description="X coordinate in pixels to scroll at.")
    y: float = Field(description="Y coordinate in pixels to scroll at.")
    delta_x: float = Field(description="Pixels to scroll horizontally.")
    delta_y: float = Field(description="Pixels to scroll vertically.")

    def __str__(self) -> str:
        return f"{self.name}(x={self.x!r}, y={self.y!r}, delta_x={self.delta_x!r}, delta_y={self.delta_y!r})"

    def describe(self, **_) -> str:
        if self.delta_x == 0:
            d = "down" if self.delta_y > 0 else "up"
            return f"at ({self.x}, {self.y}), scroll {d} by {abs(self.delta_y)} pixels"
        elif self.delta_y == 0:
            d = "right" if self.delta_x > 0 else "left"
            return f"at ({self.x}, {self.y}), scroll {d} by {abs(self.delta_x)} pixels"
        return f"at ({self.x}, {self.y}), scroll ({self.delta_x}, {self.delta_y}) pixels"


class MouseDragAndDrop(BaseModel):
    name: ClassVar[str] = "mouse_drag_and_drop"
    from_x: float = Field(..., description="X pixel to start drag.")
    from_y: float = Field(..., description="Y pixel to start drag.")
    to_x: float = Field(..., description="X pixel to release drag.")
    to_y: float = Field(..., description="Y pixel to release drag.")

    def __str__(self) -> str:
        return f"{self.name}(from_x={self.from_x!r}, from_y={self.from_y!r}, to_x={self.to_x!r}, to_y={self.to_y!r})"

    def describe(self, **_) -> str:
        return f"drag from ({self.from_x}, {self.from_y}) to ({self.to_x}, {self.to_y})"


class KeyboardType(BaseModel):
    name: ClassVar[str] = "keyboard_type"
    text: str = Field(description="The text to type.")

    def __str__(self) -> str:
        return f"{self.name}(text={self.text!r})"

    def describe(self, **_) -> str:
        return f"type {self.text!r}"


class KeyboardPress(BaseModel):
    name: ClassVar[str] = "keyboard_press"
    key: str = Field(description="The key to press, e.g. 'Enter', 'Escape', 'Control+L'.")

    def __str__(self) -> str:
        return f"{self.name}(key={self.key!r})"

    def describe(self, **_) -> str:
        return f"press {self.key} key"


class SelectAll(BaseModel):
    name: ClassVar[str] = "keyboard_press"
    key: Literal["ControlOrMeta+a"] = Field("ControlOrMeta+a")

    def __str__(self) -> str:
        return f"{self.name}(key={self.key!r})"

    def describe(self, **_) -> str:
        return "select all content"


class GeminiTypeTextAt(BaseModel):
    name: ClassVar[str] = "gemini_type_text_at"
    x: float = Field(description="X pixel coordinate to type at.")
    y: float = Field(description="Y pixel coordinate to type at.")
    text: str = Field(description="The text to type.")
    press_enter: bool = Field(default=True)
    clear_before_typing: bool = Field(default=True)

    def __str__(self) -> str:
        return f"{self.name}(x={self.x!r}, y={self.y!r}, text={self.text!r}, press_enter={self.press_enter!r}, clear_before_typing={self.clear_before_typing!r})"

    def describe(self, **_) -> str:
        parts = [f"type {self.text!r} at ({self.x}, {self.y})"]
        if self.clear_before_typing:
            parts.append("after clearing field")
        if self.press_enter:
            parts.append("then press Enter")
        return " ".join(parts)


class Goto(BaseModel):
    name: ClassVar[str] = "goto"
    url: str = Field(description="The url to navigate to.")

    def __str__(self) -> str:
        return f"{self.name}(url={self.url!r})"

    def describe(self, **_) -> str:
        return f"go to {self.url}"


class Noop(BaseModel):
    name: ClassVar[str] = "noop"
    noop_reason: Literal["loading", "captcha", "unsupported_keypress", "retrying_after_api_error"] = Field(
        description="Reason for no-op."
    )

    def __str__(self) -> str:
        return f"{self.name}(wait_ms={NOOP_WAIT_MS})"

    def describe(self, **_) -> str:
        if self.noop_reason == "loading":
            return "wait for page to finish loading"
        elif self.noop_reason == "captcha":
            return "wait for captcha to be solved"
        return f"noop({self.noop_reason})"


class SendMsgToUser(BaseModel):
    name: ClassVar[str] = "send_msg_to_user"
    msg: str = Field(description="The message to send to the user.")

    def __str__(self) -> str:
        return f"{self.name}(text={self.msg!r})"

    def describe(self, **_) -> str:
        return f"send message to user: {self.msg!r}"


class ReportInfeasible(BaseModel):
    name: ClassVar[str] = "report_infeasible"
    infeasibility_reason: str = Field(description="Reason why the task is infeasible.")

    def __str__(self) -> str:
        return f"{self.name}(reason={self.infeasibility_reason!r})"

    def describe(self, **_) -> str:
        return f"task infeasible: {self.infeasibility_reason}"


class BrowserNav(BaseModel):
    name: ClassVar[str] = "browser_nav"
    nav_type: Literal["go_back", "new_tab", "tab_focus"] = Field(
        description="Browser navigation type."
    )
    index: int = Field(description="Tab index for tab_focus, -1 otherwise.")

    def __str__(self) -> str:
        if self.nav_type == "tab_focus":
            return f"{self.nav_type}(index={self.index})"
        return f"{self.nav_type}()"

    def describe(self, **_) -> str:
        if self.nav_type == "go_back":
            return "go back to previous page"
        elif self.nav_type == "new_tab":
            return "open a new blank tab"
        return f"switch to tab {self.index}"


ALL_ACTIONS = Union[
    Click, MouseClick, MouseMove, HoverAt, MouseDragAndDrop,
    Scroll, ScrollAt, KeyboardType, KeyboardPress, GeminiTypeTextAt,
    SelectAll, Goto, Noop, BrowserNav, SendMsgToUser, ReportInfeasible,
]

AXTREE_ACTIONS = Union[
    Click, Scroll, KeyboardType, KeyboardPress, SelectAll,
    Goto, Noop, BrowserNav, SendMsgToUser, ReportInfeasible,
]


class ActionOutput(BaseModel):
    thought: str = Field(description="The reasoning behind the action.")
    action: ALL_ACTIONS = Field(description="The action to take.")

    @property
    def name(self):
        return self.action.name

    @property
    def params(self):
        return self.action.model_dump()

    def to_str(self) -> str:
        return str(self.action)

    def describe(self, axtree: dict | None = None, extra_element_properties: dict | None = None) -> str:
        return self.action.describe(axtree=axtree, extra_element_properties=extra_element_properties)


class AxtreeActionOutput(ActionOutput):
    """ActionOutput restricted to bid-based actions (for structured output parsing)."""
    action: AXTREE_ACTIONS = Field(description="The action to take.")  # type: ignore


# Axtree helpers for Click.describe()

def _node2str(node):
    role = node["role"]["value"]
    name = node["name"]["value"]
    if name == "":
        return role
    return f"{role} '{name}'"


def _get_node_from_bid(bid, axtree):
    for node in axtree.get("nodes", []):
        if node.get("molmoweb_id") == bid:
            return node
    return None


def get_node_properties(bid, axtree):
    node = _get_node_from_bid(bid, axtree)
    if node is None:
        return None
    return dict(role=node["role"]["value"], value=node["name"]["value"])
