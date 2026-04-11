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
    """Convert screenshot numpy array to PNG bytes."""
    _, png_bytes = cv2.imencode('.png', screenshot)
    return png_bytes.tobytes()

def build_initial_content(goal: str, screenshot: np.ndarray) -> Content:
    """Build the initial user message with goal and screenshot."""
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
    """Build function response content with screenshot."""
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


def scale_coordinate(coord: float, dimension: int) -> float:
    return round((coord / 1000.0) * dimension, 1)


def parse_model_response(candidate, screenshot: np.ndarray) -> tuple[str, ALL_ACTIONS, str | None, bool]:
    """
    Parse model response to extract thought, action, function call name, and safety decision.
    """
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
            # Check for safety_decision (we auto-approve but must acknowledge)
            if 'safety_decision' in args_dict:
                has_safety_decision = True
                args_dict.pop('safety_decision', None)
            
            action_obj = build_gemini_action(fc.name, args_dict, screenshot)
    
    # Default action if no function call
    # gemini doesnt have an exit action
    # so we exit when it has a thought and 
    # and no fucntion call
    if action_obj is None:
        # First send [ANSWER] with the thought, then [EXIT] on next call
        action_obj = SendMsgToUser(msg=f"[ANSWER] {thought}")
    
    return thought, action_obj, function_call_name, has_safety_decision


def build_gemini_action(name: str, args: dict[str, Any], screenshot: np.ndarray) -> ALL_ACTIONS | None:
    """
    Translate Gemini Computer Use action format to our Pydantic action objects.
    Coordinates are scaled from 0-1000 to pixel values.
    """
    height, width = screenshot.shape[:2]
    
    def scale_x(x): return scale_coordinate(x, width)
    def scale_y(y): return scale_coordinate(y, height)
    
    if name == "wait_5_seconds":
        return Noop(noop_reason="loading")
    
    elif name == "go_back":
        return BrowserNav(nav_type="go_back", index=-1)
    
    elif name == "navigate":
        return Goto(url=args.get("url", ""))
    
    elif name == "click_at":
        return MouseClick(
            x=scale_x(args.get("x", 0)),
            y=scale_y(args.get("y", 0)),
            button=args.get("button", "left"),
            click_type="single"
        )
    
    elif name == "double_click_at":
        return MouseClick(
            x=scale_x(args.get("x", 0)),
            y=scale_y(args.get("y", 0)),
            button=args.get("button", "left"),
            click_type="double"
        )
    
    elif name == "type_text_at":
        return GeminiTypeTextAt( # custom action i added
            x=scale_x(args.get("x", 0)),
            y=scale_y(args.get("y", 0)),
            text=args.get("text", ""),
            press_enter=args.get("press_enter", True),
            clear_before_typing=args.get("clear_before_typing", True),
        )
    
    elif name == "scroll_document":
        direction = str(args.get("direction", "down")).lower()
        if direction == "up":
            return Scroll(delta_x=0.0, delta_y=-float(height))
        elif direction == "down":
            return Scroll(delta_x=0.0, delta_y=float(height))
        elif direction == "left":
            return Scroll(delta_x=-float(width), delta_y=0.0)
        else:  # right
            return Scroll(delta_x=float(width), delta_y=0.0)
    
    elif name == "scroll_at":
        direction = str(args.get("direction", "down")).lower()
        magnitude = float(args.get("magnitude", 800.0))
        x = scale_x(args.get("x", 0))
        y = scale_y(args.get("y", 0))
        
        if direction == "up":
            return ScrollAt(x=x, y=y, dx=0.0, dy=-magnitude)
        elif direction == "down":
            return ScrollAt(x=x, y=y, dx=0.0, dy=magnitude)
        elif direction == "left":
            return ScrollAt(x=x, y=y, dx=-magnitude, dy=0.0)
        else:  # right
            return ScrollAt(x=x, y=y, dx=magnitude, dy=0.0)
    
    elif name == "drag_and_drop":
        return MouseDragAndDrop(
            from_x=scale_x(args.get("x", 0)),
            from_y=scale_y(args.get("y", 0)),
            to_x=scale_x(args.get("destination_x", 0)),
            to_y=scale_y(args.get("destination_y", 0)),
        )
    
    elif name == "hover_at":
        return MouseMove(
            x=scale_x(args.get("x", 0)),
            y=scale_y(args.get("y", 0)),
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
        # Excluding action below because we dont support them. 
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
        self.context: dict[str, Any] | None = None

    def reset(self):
        self.contents = []
        self.last_function_call_name = None
        self.last_had_safety_decision = False
        self.block_count = 0
        self.last_model_inputs = None
        self.sent_answer = False
        self.context = None

    def set_context(self, context: dict[str, Any] | None):
        """
        Set context from previous task for follow-up queries.
        
        Context includes:
        - last_screenshot_base64: Screenshot when previous task completed
        - last_url: URL when previous task completed
        - last_thought: Agent's last thought
        - conversation_history: List of previous queries/actions
        """
        self.context = context

    def predict_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # If we've already sent the [ANSWER], immediately return [EXIT] without API call
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
        if screenshot is None: # if missing make it a white screen
            screenshot = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        
        # Build and append content
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
            # Reset after acknowledgement is sent
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
        
        # Call Gemini API with one retry on error
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
            """Check if response has valid content."""
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
        
        # Debug: print truncated response info
        if response.candidates:
            print(f"[GeminiCUAgent] Response received with {len(response.candidates)} candidate(s)")
        
        valid, error_type = is_valid_response(response)
        
        # If invalid, retry once after sleeping
        if not valid:
            time.sleep(5)
            response, api_error = call_api()
            
            if api_error:
                return f"Predictor error: {api_error}", {}
            
            if response.candidates:
                print(f"[GeminiCUAgent] Retry response received with {len(response.candidates)} candidate(s)")
            valid, error_type = is_valid_response(response)
        
        # After retry, if still invalid, increment block count and return Noop or Exit
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
        
        # Valid response - get candidate
        candidate = response.candidates[0]
        
        self.contents.append(candidate.content)
        
        thought, action_obj, function_call_name, has_safety_decision = parse_model_response(candidate, screenshot)
        
        # Update state for next turn
        if function_call_name:
            self.last_function_call_name = function_call_name
            if has_safety_decision:
                self.last_had_safety_decision = True
        else:
            self.last_function_call_name = None
        
        # Check if this is an [ANSWER] message (exit sequence initiated)
        if isinstance(action_obj, SendMsgToUser) and action_obj.msg.startswith("[ANSWER]"):
            self.sent_answer = True
        
        # Build action output
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