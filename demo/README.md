# Local MolmoWeb demo

Minimal stack (API, UI, proxy, Postgres) to run the demo locally. ***Note that this demo code uses slightly different versions of agent and browser env code for simplicity sake.***

Based on the code that powers https://molmoweb.allen.ai/. This repo does **not** ship any inference URLs or keys—you bring your own.

## Prerequisites

- [Docker](https://www.docker.com/get-started) with Compose

## Inference (molmoweb agent)

Pick one:

| Mode | What you do |
|------|-------------|
| **modal** | Deploy or use your own Modal completion endpoint. Set `INFERENCE_MODE=modal`, `MODAL_ENDPOINT` to that URL, and `MODAL_API_KEY` if it expects a Bearer token. |
| **fastapi** | Run a compatible FastAPI inference server on your machine. Set `INFERENCE_MODE=fastapi` and `FASTAPI_ENDPOINT` to a URL the **API container** can reach (often `http://localhost:<port>/...`). |

Without a valid endpoint for the chosen mode, the backend will error when creating the agent.

## Run

```bash
cd demo
cp .env.example .env
# Edit .env: inference settings above, plus BrowserBase (and optional Gemini) — see .env.example
docker compose up --build
```

Open an incognito Chrome browser and navigate to **http://localhost:8080** (nginx → UI; `/api` and `/socket.io` → API). 

