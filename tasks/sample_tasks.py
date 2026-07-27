"""Turn each OnlineMind2Web seed task into ~50 new tasks on similar websites.

For every seed task in om2w_taxonomized.csv we take its own website plus the 4 most
similar ones from nearest_websites.csv, then ask an LLM to rewrite the task for each
target site — same difficulty, details adapted to what that site actually offers.
So "find shoes on Amazon" becomes "find flip flops on AliExpress".

Each call returns a batch of `samples_per_site` tasks at once rather than one task per
call, so the model can see its own earlier suggestions and vary the product, filter, or
location instead of repeating itself. 5 sites x 10 samples = 50 tasks per seed task.
Exact duplicates are dropped at the end.

Usage:
  python tasks/sample_tasks.py --out tasks/om2w_like_tasks.json
  python tasks/sample_tasks.py --num_seeds 5 --out /tmp/smoke.json   # small trial run
"""
import asyncio
import csv
import datetime
import json
import os

import fire
from google import genai
from jinja2 import Template
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-3-flash-preview"


class GeneratedTasks(BaseModel):
    prompts: list[str] = Field(
        description="Web navigation task instructions, 1-3 sentences each."
    )


PROMPT = Template(
    """Today's date is {{ today }}.

Write {{ n }} web navigation tasks for {{ target }}, inspired by this task from
{{ source }}: "{{ example }}"

Style reference for good tasks:
- "Find a recipe for a vegetarian lasagna with at least a four-star rating that uses zucchini."
- "Look for hotels in Sydney from April 24 to April 27, 2026 on Booking. With the Swimming Pool and Airport Shuttle filters applied, how many hotels are available?"
- "Browse the online degrees section on Coursera and list 3 Bachelor's degree programmes."

Rules:
- Match the example's difficulty — same rough number of steps, no extra requirements
- Adapt to what {{ target }} actually offers; don't just swap the website name
- Include concrete details (specific products, categories, filters, locations, quantities)
- Every task must be feasible on {{ target }} and verifiable by looking at the site
- Any dates must be in the future, within 6 months of today, or relative ("next week")
- The {{ n }} tasks must be genuinely different from each other: vary the product,
  category, filter, or location — not just the wording
"""
)


async def sample_for_site(client, seed: dict, target: str, n: int, sem) -> list[dict]:
    prompt = PROMPT.render(
        n=n,
        target=target,
        source=seed["website"],
        example=seed["confirmed_task"],
        today=datetime.date.today().strftime("%B %d, %Y"),
    )
    async with sem:
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeneratedTasks,
                    "temperature": 1.0,
                },
            )
            prompts = response.parsed.prompts
        except Exception as e:
            print(f"error on {target}: {e}")
            return []

    site_slug = target.replace("https://", "").replace("/", "").replace("www.", "")
    return [
        {
            "task_id": f"{seed['task_id']}_{site_slug}_{i}",
            "prompt": p,
            "website": target,
            "task_type": seed["taxonomy"],
            "source_website": seed["website"],
            "original_task": seed["confirmed_task"],
        }
        for i, p in enumerate(prompts)
    ]


def load_neighbors(path: str) -> dict[str, list[str]]:
    with open(path) as f:
        return {
            row["website"]: [v for k, v in row.items() if k.startswith("neighbor_") and v]
            for row in csv.DictReader(f)
        }


async def run(num_seeds, samples_per_site, workers, tasks_csv, neighbors_csv, out):
    with open(tasks_csv) as f:
        seeds = list(csv.DictReader(f))[:num_seeds]
    neighbors = load_neighbors(neighbors_csv)

    client = genai.Client()
    sem = asyncio.Semaphore(workers)
    coros = [
        sample_for_site(client, seed, target, samples_per_site, sem)
        for seed in seeds
        for target in [seed["website"]] + neighbors.get(seed["website"], [])
    ]
    print(f"{len(seeds)} seed tasks -> {len(coros)} calls x {samples_per_site} tasks each")
    batches = await tqdm.gather(*coros, desc="sampling")

    tasks, seen = [], set()
    for task in [t for batch in batches for t in batch]:
        key = (task["website"], task["prompt"].strip().lower())
        if key not in seen:
            seen.add(key)
            tasks.append(task)

    generated = sum(len(b) for b in batches)
    with open(out, "w") as f:
        json.dump(tasks, f, indent=2)
    print(
        f"{len(tasks)} tasks ({generated - len(tasks)} duplicates dropped), "
        f"{len(tasks) / max(len(seeds), 1):.1f} per seed task -> {out}"
    )


def main(
    num_seeds: int = None,
    samples_per_site: int = 10,
    workers: int = 32,
    tasks_csv: str = f"{BASE}/om2w_taxonomized.csv",
    neighbors_csv: str = f"{BASE}/nearest_websites.csv",
    out: str = f"{BASE}/om2w_like_tasks.json",
):
    """Args:
        num_seeds: Only use the first N seed tasks (default: all 299).
        samples_per_site: Tasks per (seed task, website) pair; 10 gives ~50 per seed.
        workers: Concurrent LLM calls.
        tasks_csv: Seed tasks with a taxonomy column.
        neighbors_csv: Website pairing from pair_websites.py.
        out: Output JSON path.
    """
    asyncio.run(
        run(num_seeds, samples_per_site, workers, tasks_csv, neighbors_csv, out)
    )


if __name__ == "__main__":
    fire.Fire(main)
