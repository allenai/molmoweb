from typing import Any

from google import genai
from google.genai import types
from utils.axtree import flatten_axtree_to_str
from jinja2 import Template

from agent.actions import ActionOutput
from agent.utils import AgentBase
from agent.actions import AXTREE_ACTIONS, AxtreeActionOutput

SYSTEM_MESSAGE = """You are an intelligent web assistant. You will be given a representation of the current web page with unique ids (bid) for each DOM element along with the contents of the element. Your job is to predict the immediate next mouse or keyboard interaction towards accomplishing the user specified GOAL. At each step, you will provide a `thought` behind your action and the `action` to take. When you have completed the task you must send a message to the user with content 'exit'.


# Page representation
This is an example representation of the web page:
```
[1] button 'Click me!'
[2] div 'This is a textbox'
    [3] listitem ''
```
Here [1], [2], [3] are the unique ids (bid) of the elements which are arranged hierarchically. The text following the id is the content of the element.

# IMPORTANT CONSTRAINT ON THOUGHT
The `thought` field MUST be concise and high-level.  
It MUST NOT reference implementation details such as:
- bids, element ids, or DOM structure
- HTML tags, fields, attributes, or page representation mechanics
- trees, axtrees, accessibility trees, or similar internal structures

# Navigation Guidelines

1. **Direct URL Navigation**: When you need to reach a specific website or URL, use the `goto` action directly. Do not search for websites through search engines when the URL is known or can be inferred. **Critical URL Rules**:
   - Only use URLs that are explicitly visible on the page, provided in the task, or are well-known standard URLs (e.g., homepage URLs like `https://www.edmunds.com`).
   - **NEVER fabricate, guess, or construct URLs with paths, parameters, or search terms**—if you need to find specific content on a website, navigate to the base URL first, then use the site's search or navigation features.
   - For multi-step tasks like "Navigate to X and search for Y", go to the base URL (e.g., `https://www.x.com`) first, then perform the search using the website's interface.
   - Correct: `goto(url="https://www.edmunds.com")` then use the site's search
   - Correct: `goto(url="https://www.google.com")`
   - Wrong: `goto(url="https://www.edmunds.com/2024-toyota-camry/overview/")` — do not construct paths
   - Wrong: `goto(url="https://example.com/search?q=something")` — do not add query parameters
   - Wrong: `goto(url="search for recipes")` — the url field must contain only a valid URL

2. **Text Input**: Always click on an input field before typing. Ensure the element is focused before sending keystrokes.

3. **Element Interaction**: Only interact with elements that support the intended action (e.g., only click clickable elements).

4. **Scrolling**: When content is not visible, scroll to reveal it. By default, scroll 100% of the viewport height (720px) to efficiently navigate through pages. Use smaller scroll increments only when precision is required (e.g., to avoid scrolling past a specific element).

5. **Information Retrieval**: If the goal requires finding specific information, locate and extract that information, then report it to the user. If the goal is to navigate to a page or complete an action, do so without necessarily reporting back.

6. **Tab Management**: Avoid opening new tabs or navigating backward unless strictly necessary for the task.

7. **Efficiency**: Most tasks can be completed in fewer than 20 steps. Prioritize direct, efficient actions.

8. **Cloudflare/Captcha Pages**: If you encounter a Cloudflare security check, captcha, or "Please verify you are human" page, use the `noop` action and wait. Do NOT report being blocked—simply wait for the page to resolve.

# Success Criteria

The task is complete when:
- Required information has been found and communicated to the user, OR
- The requested navigation or action has been successfully performed

# Termination Protocol

**To end a task, you must perform exactly 2 actions in sequence:**

**Action 1 - Send your answer:**
```
send_msg_to_user(text='[ANSWER] <your single answer here>')
```

**Action 2 - Send exit signal (next action after answer):**
```
send_msg_to_user(text='[EXIT]')
```

**Rules:**
- The `[ANSWER]` message must contain exactly ONE concise answer—no multiple options, no explanations, no error messages.
- The `[EXIT]` message must be sent as a SEPARATE action AFTER the answer, containing ONLY `[EXIT]`.
- Never combine `[ANSWER]` and `[EXIT]` in the same message.
- If you cannot find the exact answer, provide your single best guess.
- For tasks that don't require a textual response, send `send_msg_to_user(text='[ANSWER] Done')` then `send_msg_to_user(text='[EXIT]')`.

# Reasoning Requirements

You must provide dense, comprehensive reasoning with every action. Your thought process should explicitly address ALL of the following:

1. **Goal Decomposition**: Restate the original goal. Break it down into sub-tasks or requirements. List each component that must be satisfied for full completion.

2. **Progress Assessment**: For each sub-task or requirement identified above, explicitly state whether it has been completed, is in progress, or has not yet been started. What evidence on the current page confirms completion of any sub-tasks?

3. **Gap Analysis**: What specific aspects of the goal remain unfinished? What information is still missing? What actions have not yet been taken that are required?

4. **Current Page Analysis**: What relevant elements, information, or affordances are visible? How does the current page state relate to what you need to accomplish next?

5. **Action Selection**: Given the gap between current state and goal completion, why is this specific action the optimal next step? What alternatives were considered and why were they rejected?

6. **Expected Outcome**: What will this action achieve? How will it move you closer to completing the remaining requirements?

Your reasoning should be thorough and information-dense. Avoid vague statements. Be specific about what has been done, what remains, and why your chosen action addresses the most critical remaining gap.
Reminder that the thought field must avoid any mentions of bids, DOM structure, HTML tags or axtree. The thought must be a brief and high level plan.
"""

AVAILABLE_WEBSITES_TEMPLATE = Template(
    """# WEBSITES AVAILABLE
The tasks require one of the following websites hosted at these urls:
{% for title, url in available_websites.items() -%}
    {{ title }}: {{ url }}
{% endfor %}
"""
)


USER_MSG_TEMPLATE = Template(
    """# GOAL
{{ task_description }}

# PREVIOUS ACTIONS
{% for action in past_actions -%}
## Step {{ loop.index }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action_str'] }}
ACTION DESCRIPTION: {{ action['action_description'] }}
{% endfor %}

# CURRENT OBSERVATION

## ALL PAGES / TABS OPEN IN THE BROWSER
{% for title, url in open_pages_titles_and_urls -%}
    Page {{ loop.index-1 }}: {{ title }} | {{ url }}
{% endfor %}

## CURRENTLY ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

## AXTREE OF THE CURRENTLY ACTIVE PAGE
{{ axtree_str }}

## ERRORS FROM THE LAST ACTION (IF ANY)
{{ last_action_error }}
"""
)


def create_llm_action_predictor(
    response_format: type[ActionOutput] = AxtreeActionOutput,
    model: str = "gemini-3-flash-preview",
):
    client = genai.Client()
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_format,
    )

    def get_action_output_from_llm(
        system_message: str, user_message: str
    ) -> tuple[AXTREE_ACTIONS, str | None]:
        response = client.models.generate_content(
            model=model,
            contents=[user_message],
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                response_mime_type="application/json",
                response_schema=response_format,
            ),
        )
        
        raw = response.text
        parsed = response_format.model_validate_json(raw)
        return parsed, raw

    return get_action_output_from_llm


def get_axtree_str(obs):
    return flatten_axtree_to_str(
        obs["axtree_object"],
        extra_properties=obs["extra_element_properties"],
        with_visible=False,
        filter_visible_only=True,
        filter_with_bid_only=True,
        with_clickable=True,
        skip_generic=True,
    )


def get_user_message(
    obs: dict[str, Any],
    past_actions: list[dict[str, Any]],
    past_urls: list[str],
    available_websites: dict[str, str] | None = None,
    website_guidelines: str | None = None,
    axtree_str: str | None = None,
) -> str:
    msg = ""
    if available_websites is not None:
        msg = AVAILABLE_WEBSITES_TEMPLATE.render(
            available_websites=available_websites
        )

    last_action_error = "The action was successful with no error."
    if obs["last_action_error"] != "":
        if "TimeoutError" not in obs["last_action_error"]:
            last_action_error = obs["last_action_error"]

    if axtree_str is None:
        axtree_str = get_axtree_str(obs)

    page_index = int(obs["active_page_index"][0])
    user_message = msg + USER_MSG_TEMPLATE.render(
        axtree_str=axtree_str,
        open_pages_titles_and_urls=zip(
            obs["open_pages_titles"], obs["open_pages_urls"]
        ),
        page_title=obs["open_pages_titles"][page_index],
        page_url=obs["open_pages_urls"][page_index],
        page_index=page_index,
        url=obs["url"],
        task_description=obs["goal"],
        past_actions=past_actions[-10:],
        past_urls=past_urls[-10:],
        last_action_error=last_action_error,
    )

    if website_guidelines is not None:
        user_message += f"\n# Website specific guidelines\n{website_guidelines}"

    return user_message


class GeminiAxtreeAgent(AgentBase):
    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        llm_response_format: type[ActionOutput] = AxtreeActionOutput,
        system_message: str = SYSTEM_MESSAGE,
        website_guidelines: str | None = None,
    ):
        self.model = model
        self.llm_action_predictor = create_llm_action_predictor(
            response_format=llm_response_format,
            model=model,
        )
        self.system_message = system_message
        self.website_guidelines = website_guidelines
        self.past_actions: list[dict[str, Any]] = []
        self.past_user_messages: list[str] = []
        self.past_urls: list[str] = []
        self.last_model_inputs: dict[str, Any] | None = None

    def reset(self):
        self.past_actions: list[dict[str, Any]] = []
        self.past_user_messages: list[str] = []
        self.past_urls: list[str] = []
        self.last_model_inputs: dict[str, Any] | None = None

    def predict_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        axtree_str = get_axtree_str(obs)
        user_message = get_user_message(
            obs,
            self.past_actions,
            self.past_urls,
            website_guidelines=self.website_guidelines,
            axtree_str=axtree_str,
        )
        action_output, raw_text = self.llm_action_predictor(
            self.system_message, user_message
        )
        self.last_model_inputs = {
            "axtree_str": axtree_str,
            "system_message": self.system_message,
            "user_message": user_message,
            "page_index": int(obs["active_page_index"][0]),
            "url": obs["url"],
            "open_pages_titles": obs["open_pages_titles"],
            "open_pages_urls": obs["open_pages_urls"],
        }
        self.past_user_messages.append(user_message)
        self.past_actions.append(
            {
                "action_output": action_output,
                "thought": action_output.thought,
                "action_str": action_output.to_str(),
                "action_description": action_output.describe(
                    axtree=obs["axtree_object"],
                    extra_element_properties=obs["extra_element_properties"],
                ),
                "raw_output": raw_text,
            }
        )
        self.past_urls.append(obs["url"])
        return raw_text, self.past_actions[-1]

    def get_last_model_inputs(self) -> dict[str, Any] | None:
        return self.last_model_inputs
