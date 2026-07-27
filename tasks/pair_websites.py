"""Pair each OnlineMind2Web website with the 4 most similar websites in the seed set.

Asks an LLM to pick, for every website appearing in om2w_taxonomized.csv, the sites a
user would consider alternatives or competitors. Writes nearest_websites.csv, which
sample_tasks.py reads to decide where to re-target each seed task.

The output is committed, so this only needs re-running when the seed set changes.

Usage:
  python tasks/pair_websites.py
"""
import asyncio
import csv
import os

import fire
from google import genai
from jinja2 import Template
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-3-flash-preview"


class NearestWebsites(BaseModel):
    neighbors: list[str] = Field(
        description="Website origins from the candidate list, most to least similar."
    )


PROMPT = Template(
    """Pick the {{ n }} websites from the candidate list most similar to the query website.
"Similar" means a user would consider them alternatives or competitors.

Examples: alaskaair.com -> delta.com, united.com, aa.com, southwest.com
          amazon.com -> walmart.com, target.com, bestbuy.com, costco.com

Query website: {{ query }}

Candidates:
{% for c in candidates %}{{ c }}
{% endfor %}
Return EXACTLY {{ n }} origins from the candidate list, most similar first, excluding the query.
"""
)


async def find_neighbors(client, query: str, candidates: list[str], n: int, sem) -> list[str]:
    prompt = PROMPT.render(query=query, candidates=[c for c in candidates if c != query], n=n)
    async with sem:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": NearestWebsites},
        )
    return [w for w in response.parsed.neighbors if w in candidates and w != query][:n]


async def run(n: int, workers: int, tasks_csv: str, out_csv: str):
    with open(tasks_csv) as f:
        websites = sorted({row["website"] for row in csv.DictReader(f)})
    print(f"{len(websites)} websites, {n} neighbors each")

    client = genai.Client()
    sem = asyncio.Semaphore(workers)
    coros = [find_neighbors(client, w, websites, n, sem) for w in websites]
    neighbors = await tqdm.gather(*coros, desc="pairing")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["website"] + [f"neighbor_{i + 1}" for i in range(n)])
        for website, neighs in zip(websites, neighbors):
            writer.writerow([website] + neighs + [""] * (n - len(neighs)))
    print(f"wrote {out_csv}")


def main(
    n: int = 4,
    workers: int = 32,
    tasks_csv: str = f"{BASE}/om2w_taxonomized.csv",
    out_csv: str = f"{BASE}/nearest_websites.csv",
):
    """Args:
        n: Neighbors per website.
        workers: Concurrent LLM calls.
        tasks_csv: Seed tasks; its `website` column defines both queries and candidates.
        out_csv: Output pairing table.
    """
    asyncio.run(run(n, workers, tasks_csv, out_csv))


if __name__ == "__main__":
    fire.Fire(main)
