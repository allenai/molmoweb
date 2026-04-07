"""Judge for the DeepShop benchmark."""
from typing import Literal

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from benchmarks.judges.utils import encode_image

DEEPSHOP_SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:
1. Web Task Instruction: A clear and precise natural language directive that specifies an
online shopping activity to be executed. The instruction may involve locating products that
meet certain attribute requirements (e.g., color, size, brand), applying specific search filters
(e.g., price range, customer ratings, availability), or fulfilling user-defined sorting preferences
(e.g., lowest price, newest arrivals, best sellers). Tasks may also include verifying product
details, comparing offers, or checking for shipping and return policies, depending on the
scenario.
2. Result Screenshots: This is a visual representation of the screen showing the result or
intermediate state of performing a web task. It serves as visual proof of the actions taken in
response to the instruction.
3. Result Response: This is a textual response obtained after the execution of the web task.
It serves as textual result in response to the instruction.
-- You DO NOT NEED to interact with web pages or perform actions such as conducting
searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the
screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction
against the outcome depicted in the screenshot and in the response, evaluating whether the
actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the
garage and summarizing the review. Failing to complete either task, such as not providing a
summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at
the end of web browsing, and there may be discrepancies between the text and the
screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of
the screenshot prevails, 2) The content in the Result response is not mentioned on the
screenshot, choose to believe the content.
You should elaborate on how you arrived at your final evaluation and then provide a
definitive verdict on whether the task has been successfully accomplished, either as
'SUCCESS' or 'NOT SUCCESS'."""


class DeepShopVerdict(BaseModel):
    thought: str = Field(description="Reasoning behind the verdict")
    verdict: Literal["SUCCESS", "NOT SUCCESS"] = Field(
        description="Final verdict for whether the web assistant completed the task or not."
    )


class DeepShopVerdictNormalized(BaseModel):
    thought: str
    verdict: Literal["SUCCESS", "FAILURE"]


def get_verdict_deepshop(
    task: str, answer: str, screenshots: list[Image.Image]
) -> DeepShopVerdictNormalized:
    client = OpenAI()

    user_content = [
        {"type": "text", "text": f"TASK:\n{task}\nResult Response:\n{answer}\n{len(screenshots)} screenshots"},
    ]
    for i, screenshot in enumerate(screenshots):
        b64 = encode_image(screenshot)
        user_content.append({"type": "text", "text": f"Step {i+1}"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": DEEPSHOP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=DeepShopVerdict,
    )
    raw = response.choices[0].message.parsed
    verdict = "FAILURE" if raw.verdict == "NOT SUCCESS" else raw.verdict
    return DeepShopVerdictNormalized(thought=raw.thought, verdict=verdict)
