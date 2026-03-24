from typing import Any
import time
import numpy as np
import cv2
from google import genai
from google.genai import types
from google.genai.types import Content, Part

from agent.actions import (
    ActionOutput,
    MouseDragAndDrop,
    ALL_ACTIONS,
    MouseClick,
    MouseMove,
    Scroll,
    ScrollAt,
    GeminiTypeTextAt,
    Goto,
    BrowserNav,
    Noop,
    SendMsgToUser,
)
from agent.utils import AgentBase

def screenshot_to_png_bytes(screenshot: np.ndarray) -> bytes:
    _, png_bytes = cv2.imencode('.png', screenshot)
    return png_bytes.tobytes()

def build_initial_content(goal: str, screenshot: np.ndarray) -> Content:
    screenshot_bytes = screenshot_to_png_bytes(screenshot)
    return Content(
        role="user",
        parts=[
            Part(text=goal),
            Part.from_bytes(data=screenshot_bytes, mime_type='image/png'),
        ]
    )

def build_function_response_content(
    function_name: str,
    screenshot: np.ndarray,
    url: str,
    error: str | None = None,
    acknowledge_safety: bool = False,
) -> Content:
    screenshot_bytes = screenshot_to_png_bytes(screenshot)
    
    response_data = {"url": url}
    if error:
        response_data["error"] = error
    if acknowledge_safety:
        response_data["safety_acknowledgement"] = "true"
    
    function_response = types.FunctionResponse(
        name=function_name,
        response=response_data,
        parts=[types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(
                mime_type="image/png",
                data=screenshot_bytes
            )
        )]
    )
    
    return Content(
        role="user",
        parts=[Part(function_response=function_response)]
    )


def _scale_1000_to_px(coord: float, dim: int) -> float:
    """Convert Gemini's 0-999 normalized coordinate to pixel value."""
    return round((coord / 1000.0) * dim, 1)


def _scale_1000_to_coord(coord: float, dim: int) -> float:
    """Convert Gemini's 0-999 normalized coordinate to pixel, clamped inside viewport."""
    px = round((coord / 1000.0) * dim, 1)
    return max(1.0, min(px, dim - 2.0))


def parse_model_response(candidate, screenshot: np.ndarray) -> tuple[str, ALL_ACTIONS, str | None, bool]:
    thought = ""
    action_obj = None
    function_call_name = None
    has_safety_decision = False
    
    for part in candidate.content.parts:
        if part.text:
            thought = part.text
        if part.function_call:
            fc = part.function_call
            function_call_name = fc.name
            
            args_dict = dict(fc.args or {})
            if 'safety_decision' in args_dict:
                has_safety_decision = True
                args_dict.pop('safety_decision', None)
            
            action_obj = build_gemini_action(fc.name, args_dict, screenshot)
    
    if action_obj is None:
        action_obj = SendMsgToUser(msg=f"[ANSWER] {thought}")
    
    return thought, action_obj, function_call_name, has_safety_decision


def build_gemini_action(name: str, args: dict[str, Any], screenshot: np.ndarray) -> ALL_ACTIONS | None:
    h, w = screenshot.shape[:2]

    def sx(x): return _scale_1000_to_coord(x, w)
    def sy(y): return _scale_1000_to_coord(y, h)

    if name == "wait_5_seconds":
        return Noop(noop_reason="loading")

    elif name == "go_back":
        return BrowserNav(nav_type="go_back", index=-1)

    elif name == "navigate":
        return Goto(url=args.get("url", ""))

    elif name == "click_at":
        return MouseClick(
            x=sx(args.get("x", 0)),
            y=sy(args.get("y", 0)),
            button=args.get("button", "left"),
            click_type="single"
        )

    elif name == "double_click_at":
        return MouseClick(
            x=sx(args.get("x", 0)),
            y=sy(args.get("y", 0)),
            button=args.get("button", "left"),
            click_type="double"
        )

    elif name == "type_text_at":
        return GeminiTypeTextAt(
            x=sx(args.get("x", 0)),
            y=sy(args.get("y", 0)),
            text=args.get("text", ""),
            press_enter=args.get("press_enter", True),
            clear_before_typing=args.get("clear_before_typing", True),
        )

    elif name == "scroll_document":
        direction = str(args.get("direction", "down")).lower()
        if direction == "up":
            return Scroll(delta_x=0.0, delta_y=-float(h))
        elif direction == "down":
            return Scroll(delta_x=0.0, delta_y=float(h))
        elif direction == "left":
            return Scroll(delta_x=-float(w), delta_y=0.0)
        else:
            return Scroll(delta_x=float(w), delta_y=0.0)

    elif name == "scroll_at":
        direction = str(args.get("direction", "down")).lower()
        magnitude = float(args.get("magnitude", 800))
        x, y = sx(args.get("x", 0)), sy(args.get("y", 0))
        px_mag_x = _scale_1000_to_px(magnitude, w)
        px_mag_y = _scale_1000_to_px(magnitude, h)

        if direction == "up":
            return ScrollAt(x=x, y=y, delta_x=0.0, delta_y=-px_mag_y)
        elif direction == "down":
            return ScrollAt(x=x, y=y, delta_x=0.0, delta_y=px_mag_y)
        elif direction == "left":
            return ScrollAt(x=x, y=y, delta_x=-px_mag_x, delta_y=0.0)
        else:
            return ScrollAt(x=x, y=y, delta_x=px_mag_x, delta_y=0.0)
    
    elif name == "drag_and_drop":
        return MouseDragAndDrop(
            from_x=sx(args.get("x", 0)),
            from_y=sy(args.get("y", 0)),
            to_x=sx(args.get("destination_x", 0)),
            to_y=sy(args.get("destination_y", 0)),
        )
    
    elif name == "hover_at":
        return MouseMove(
            x=sx(args.get("x", 0)),
            y=sy(args.get("y", 0)),
        )
    
    return None

class GeminiCUAgent(AgentBase):

    def __init__(
        self,
        model: str = "gemini-2.5-computer-use-preview-10-2025",
        excluded_predefined_functions: list[str] | None = None,
        max_blocks_before_terminate: int = 10,
    ):
        self.client = genai.Client()
        self.model = model
        self.excluded_predefined_functions = excluded_predefined_functions or [
            "open_web_browser", "go_forward", "search", "key_combination",
        ]
        self.max_blocks_before_terminate = max_blocks_before_terminate
        self.config = genai.types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=self.excluded_predefined_functions,
                    )
                )
            ],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
        self.contents: list[Content] = []
        self.last_function_call_name: str | None = None
        self.last_had_safety_decision: bool = False
        self.block_count: int = 0
        self.last_model_inputs: dict[str, Any] | None = None
        self.sent_answer: bool = False

    def reset(self):
        self.contents = []
        self.last_function_call_name = None
        self.last_had_safety_decision = False
        self.block_count = 0
        self.last_model_inputs = None
        self.sent_answer = False

    def predict_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.sent_answer:
            exit_action = SendMsgToUser(msg="[EXIT]")
            action_output = ActionOutput(action=exit_action, thought="")
            return "Forced exit after answer", {
                "action_output": action_output,
                "thought": "The previous action sent a message to the user with the [ANSWER] prefix, indicating the question has been answered. Since the answer has been submitted, I should now exit and finish the trajectory.",
                "action_str": action_output.to_str(),
                "action_description": action_output.to_str(),
            }
        
        screenshot = obs.get("screenshot")
        if screenshot is None:
            screenshot = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        
        if len(self.contents) == 0:
            content = build_initial_content(obs["goal"], screenshot)
        elif self.last_function_call_name:
            content = build_function_response_content(
                self.last_function_call_name,
                screenshot,
                obs["url"],
                obs.get("last_action_error"),
                acknowledge_safety=self.last_had_safety_decision,
            )
            self.last_had_safety_decision = False
        else:
            content = Content(
                role="user",
                parts=[
                    Part(text="Current browser state:"),
                    Part.from_bytes(data=screenshot_to_png_bytes(screenshot), mime_type='image/png'),
                ]
            )
        self.contents.append(content)
        
        def call_api():
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=self.contents,
                    config=self.config,
                )
                return resp, None
            except Exception as e:
                return None, str(e)
        
        def is_valid_response(resp):
            if not resp:
                return False, "None response"
            if not resp.candidates or len(resp.candidates) == 0:
                is_blocked = (hasattr(resp, 'prompt_feedback') and 
                             resp.prompt_feedback and hasattr(resp.prompt_feedback, 'block_reason') and 
                             resp.prompt_feedback.block_reason)
                return False, "blocked" if is_blocked else "empty_candidates"
            candidate = resp.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return False, "empty_content"
            return True, None
        
        response, api_error = call_api()
        
        if api_error:
            return f"Predictor error: {api_error}", {}
        
        valid, error_type = is_valid_response(response)
        
        if not valid:
            time.sleep(5)
            response, api_error = call_api()
            
            if api_error:
                return f"Predictor error: {api_error}", {}
            valid, error_type = is_valid_response(response)
        
        if not valid:
            self.block_count += 1
            should_terminate = self.block_count >= self.max_blocks_before_terminate
            self.last_function_call_name = None
            
            if should_terminate:
                msg = "[EXIT]"
                thought = f"Terminated due to repeated errors ({error_type})"
                exit_action = SendMsgToUser(msg=msg)
                action_output = ActionOutput(action=exit_action, thought=thought)
                return str(response) if response else "None", {
                    "action_output": action_output,
                    "thought": thought,
                    "action_str": action_output.to_str(),
                    "action_description": action_output.to_str(),
                }
            else:
                thought = f"API error ({error_type}) persisted after retry, returning Noop for new screenshot"
                noop_action = Noop(noop_reason="retrying_after_api_error")
                action_output = ActionOutput(action=noop_action, thought=thought)
                return f"Noop after retry failed: {error_type}", {
                    "action_output": action_output,
                    "thought": thought,
                    "action_str": action_output.to_str(),
                    "action_description": action_output.to_str(),
                }
        
        candidate = response.candidates[0]
        
        self.contents.append(candidate.content)
        
        thought, action_obj, function_call_name, has_safety_decision = parse_model_response(candidate, screenshot)
        
        if function_call_name:
            self.last_function_call_name = function_call_name
            if has_safety_decision:
                self.last_had_safety_decision = True
        else:
            self.last_function_call_name = None
        
        if isinstance(action_obj, SendMsgToUser) and action_obj.msg.startswith("[ANSWER]"):
            self.sent_answer = True
        
        action_output = ActionOutput(action=action_obj, thought=thought)
        
        self.last_model_inputs = {
            "contents_count": len(self.contents),
            "model": self.model,
            "thought": thought,
        }
        
        return str(candidate.content), {
            "action_output": action_output,
            "thought": thought,
            "action_str": action_output.to_str(),
            "action_description": action_output.to_str(),
        }

    def get_last_model_inputs(self) -> dict[str, Any] | None:
        return self.last_model_inputs
