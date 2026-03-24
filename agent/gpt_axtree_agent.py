import os
from typing import Any

from openai import OpenAI
from utils.axtree import flatten_axtree_to_str
from jinja2 import Template

from agent.actions import ActionOutput
from agent.utils import AgentBase
from agent.actions import AXTREE_ACTIONS, AxtreeActionOutput
from agent.gemini_axtree_agent import SYSTEM_MESSAGE

GPT_AXTREE_MODEL = os.environ.get("GPT_AXTREE_MODEL", "gpt-5")

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


_client = OpenAI()


def create_llm_action_predictor(
    response_format: type[ActionOutput] = AxtreeActionOutput,
):
    def get_action_output_from_llm(
        system_message: str, user_message: str
    ) -> tuple[ActionOutput, str | None]:
        response = _client.beta.chat.completions.parse(
            model=GPT_AXTREE_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            response_format=response_format,
        )
        message = response.choices[0].message
        return message.parsed, message.content

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


class GPTAxtreeAgent(AgentBase):
    def __init__(
        self,
        llm_response_format: type[ActionOutput] = AxtreeActionOutput,
        system_message: str = SYSTEM_MESSAGE,
        website_guidelines: str | None = None,
    ):
        print(f"GPTAxtreeAgent using model: {GPT_AXTREE_MODEL}")
        self.llm_action_predictor = create_llm_action_predictor(
            response_format=llm_response_format
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
