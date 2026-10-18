# Deep Agents research quickstart — free

A [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) research
agent running on **free tiers only**. Started from the
[official quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart),
adapted to work without paid API keys.

## Current setup

| Component | Choice | Cost |
| --- | --- | --- |
| Model | `gemini-3.7-flash` + 2 fallbacks (Gemini free tier) | free, rate-limited |
| Web search | Tavily → DuckDuckGo/Bing fallback | free tier |
| Framework | `deepagents` | free |

## What the agent gets

Built into `create_deep_agent`:

- **File system tools** — `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **`task`** — spawns subagents in isolated context windows
- **Context management** — offloads large tool results to files

Added here:

- **`internet_search`** — web search across Tavily and the keyless engines, with
  retries and graceful failure

## Use it

An `agents` command is installed at `~/.local/bin/agents` (already on your
PATH), so you can call it from any directory:

```bash
agents "what is langgraph?"      # ask anything
agents                           # no arguments: it prompts you
agents --check                   # verify setup, no model calls
agents --ping                    # test the model with one tiny call
agents --help                    # full usage
```

Output streams each tool call, so you can watch the agent search and delegate.

<details>
<summary>Without the wrapper</summary>

```bash
cd ~/deepagents-quickstart
uv run research_agent.py "your question here"
```

</details>

## Where reports go

Nothing is lost — every run is written to markdown:

```
~/research-reports/2026-10-18-124746-what-is-langgraph.md
```

Each file records the question, the timestamp, the model used, and the full
report. Change the location with `REPORTS_DIR` in `.env`.

## Free-tier gotchas (measured, not assumed)

These were all verified by testing, and each one changed the design:

- **Gemini's built-in `google_search` grounding is NOT in the free tier.** It
  returns `429 RESOURCE_EXHAUSTED` while plain function tools work fine. That's
  why search falls back to DuckDuckGo.
- **DuckDuckGo rate-limits under bursty agent traffic.** Agents fire several
  searches in seconds. The tool retries and rotates engines, and returns an
  `{"error": ...}` dict instead of raising — a raised exception kills the run.
- **Only some `ddgs` backends work.** Verified: `duckduckgo` and `bing` return
  results. `brave`, `mojeek`, `startpage`, `google`, and `wikipedia` return
  "No results found"; `yahoo` times out. `SEARCH_BACKENDS` defaults to the two
  that work.
- **`langchain-google-genai` returns content as a list of blocks**, not a
  string. Anything reading `message.content` directly must handle both.

## Model options

```dotenv
GOOGLE_MODEL=gemini-3.6-flash
```

Other flash models on the free tier: `gemini-3.5-flash-lite` (higher limits),
`gemini-3.5-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`. Verify before
relying on one:

```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from google import genai
print(genai.Client(api_key=os.environ['GOOGLE_API_KEY']).models.generate_content(model='MODEL', contents='hi').text)
"
```

## Local fallback (no internet, no key)

```dotenv
PROVIDER=ollama
OLLAMA_MODEL=gemma4:26b
```

Installed and tool-calling verified: `gemma4:26b` (17 GB) and `gemma4:latest`
(9.6 GB). Slower than Gemini and weaker at tool discipline — a local model once
tried `read_file` on a URL instead of searching.

## Why not opencode / DeepSeek V4

- **opencode free tier → `403 FreeTierError`**: free models are restricted to
  the opencode app and refuse external scripts.
- **opencode paid tier → `402`**: needs account credits. Beware $20 auto-reload
  below a $5 balance.
- **DeepSeek V4.1 Flash / V4 Pro cannot run locally.** MIT-licensed and free to
  download, but 552B / 1.6T backbone params — roughly 380 GB–800 GB even at
  4-bit, against **48 GB RAM** and **~22 GB free disk** here. Ollama serves them
  only as `:cloud` tags (hosted, account required).

## Optional: LangSmith tracing

Off by default.

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-...
```
