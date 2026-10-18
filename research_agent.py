"""Deep Agents research agent — runs free.

Follows https://docs.langchain.com/oss/python/deepagents/quickstart

Current setup (see .env):
  - model  : Gemini free tier (gemini-3.7-flash) with fallback models
  - search : Tavily if TAVILY_API_KEY is set, else DuckDuckGo/Bing (no key)

Other providers via the PROVIDER env var:
  google   (active)  Gemini free tier   - needs GOOGLE_API_KEY, free
  ollama             local Ollama model - free, no signup, fully offline
  opencode           opencode Console   - needs account credits

Usage — the `agents` command wraps all of this:
    agents "your question here"     # ask anything
    agents                          # prompts you for a question
    agents --check                  # build the agent, no model calls
    agents --ping                   # one tiny call to test the model

Or directly:
    uv run research_agent.py "your question here"
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# langchain-google-genai emits a wall of harmless schema warnings; silence them.
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

# duckduckgo (free, no key) | google (built-in grounding — NOT on the free tier)
SEARCH = os.getenv("SEARCH", "duckduckgo").lower()

# DuckDuckGo rate-limits under bursty agent traffic, so fall back to Bing.
# Verified working backends: duckduckgo, bing, auto.
# (brave/mojeek/startpage/google/wikipedia return "No results"; yahoo times out.)
SEARCH_BACKENDS = [
    b.strip()
    for b in os.getenv("SEARCH_BACKENDS", "duckduckgo,bing").split(",")
    if b.strip()
]

# Optional: Tavily (free tier ~1,000 searches/month, https://tavily.com).
# When set it's tried first, because it's far more reliable than the keyless
# engines — those then become the fallback instead of the only option.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# --- local Ollama (default, free) --------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "900"))

# --- Google Gemini free tier ---------------------------------------------------
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.7-flash")

# Free-tier flash models get congested ("503 high demand") in waves. If the
# primary is overloaded, retry it and then fall through to these.
GOOGLE_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "GOOGLE_FALLBACKS", "gemini-3.5-flash-lite,gemini-3.8-flash"
    ).split(",")
    if m.strip()
]

# Errors worth retrying rather than crashing on.
TRANSIENT_MARKERS = (
    "503",
    "unavailable",
    "high demand",
    "overloaded",
    "429",
    "resource_exhausted",
    "rate limit",
    "deadline",
    "timeout",
)
RETRIES_PER_MODEL = int(os.getenv("RETRIES_PER_MODEL", "2"))

# Where finished reports are written as markdown.
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "~/research-reports")).expanduser()

# --- opencode Console (paid) ---------------------------------------------------
OPENCODE_BASE_URL = os.getenv(
    "OPENCODE_BASE_URL", "https://opencode.ai/inference/openai/v1"
)
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")


# --- Step 3: create a search tool --------------------------------------------
def _search_sources() -> list[str]:
    """Search sources that will be tried, in order."""
    sources = list(SEARCH_BACKENDS)
    if TAVILY_API_KEY:
        sources.insert(0, "tavily")
    return sources


def _tavily(query: str, max_results: int) -> list[dict]:
    """Search via Tavily (needs TAVILY_API_KEY)."""
    from tavily import TavilyClient

    resp = TavilyClient(api_key=TAVILY_API_KEY).search(
        query, max_results=max_results
    )
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content"),
        }
        for r in resp.get("results", [])
    ]


# Search results are cached so a retry doesn't re-hit the network. When a run
# dies on a transient model error it restarts from scratch, and without this it
# would repeat every search — burning quota and time for nothing. Failures are
# cached too, but briefly: a search that just timed out for 50s should not be
# retried immediately, yet should still recover quickly once the backend is back.
_SEARCH_CACHE: dict[tuple[str, int], tuple[float, object, bool]] = {}
_SEARCH_TTL = float(os.getenv("SEARCH_CACHE_TTL", "300"))
_SEARCH_ERROR_TTL = float(os.getenv("SEARCH_ERROR_CACHE_TTL", "60"))


def _cache_get(key: tuple[str, int]):
    hit = _SEARCH_CACHE.get(key)
    if not hit:
        return None
    ts, value, ok = hit
    ttl = _SEARCH_TTL if ok else _SEARCH_ERROR_TTL
    return value if (time.time() - ts) < ttl else None


def internet_search(query: str, max_results: int = 5):
    """Run a web search.

    Prefers Tavily when TAVILY_API_KEY is set, then falls back to the keyless
    ddgs engines. Retries because free backends rate-limit under bursty agent
    traffic. Never raises — returns an {"error": ...} dict so the agent can adapt
    instead of the whole run crashing.
    """
    key = (query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    errors = []

    if TAVILY_API_KEY:
        for attempt in range(2):
            try:
                results = _tavily(query, max_results)
                if results:
                    _SEARCH_CACHE[key] = (time.time(), results, True)
                    return results
                errors.append("tavily: no results")
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                errors.append(f"tavily: {type(exc).__name__}: {exc}")
            time.sleep(1.0 + attempt)

    from ddgs import DDGS

    for backend in SEARCH_BACKENDS:
        for attempt in range(2):
            try:
                raw = DDGS().text(query, max_results=max_results, backend=backend)
                if raw:
                    results = [
                        {
                            "title": r.get("title"),
                            "url": r.get("href"),
                            "snippet": r.get("body"),
                        }
                        for r in raw
                    ]
                    _SEARCH_CACHE[key] = (time.time(), results, True)
                    return results
                errors.append(f"{backend}: no results")
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                errors.append(f"{backend}: {type(exc).__name__}: {exc}")
            time.sleep(1.0 + attempt)

    failure = {
        "error": f"All search sources failed ({', '.join(_search_sources())}). "
        f"Last error: {errors[-1] if errors else 'unknown'}"
    }
    _SEARCH_CACHE[key] = (time.time(), failure, False)
    return failure


# --- Step 4: create the deep agent -------------------------------------------
research_instructions = """You are an expert researcher. Your job is to conduct \
thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering \
information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the \
max number of results to return.
"""


def build_model(model_override: str | None = None):
    """Return the chat model for the selected provider."""
    if PROVIDER == "ollama":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores this; required by the SDK
            timeout=OLLAMA_TIMEOUT,
        )

    if PROVIDER == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit(
                "PROVIDER=google needs GOOGLE_API_KEY.\n"
                "Get a free key at https://aistudio.google.com/apikey\n"
                "then add GOOGLE_API_KEY=... to .env\n"
                "(also run: uv add langchain-google-genai)"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            sys.exit("Run: uv add langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=model_override or GOOGLE_MODEL, api_key=api_key
        )

    if PROVIDER == "opencode":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            sys.exit(
                "PROVIDER=opencode needs OPENCODE_API_KEY.\n"
                "Note: opencode's FREE models are restricted to their own app and\n"
                "return 403 from any external script. This path needs paid credits.\n"
                "Create a key at https://opencode.ai/console"
            )
        return ChatOpenAI(
            model=OPENCODE_MODEL,
            base_url=OPENCODE_BASE_URL,
            api_key=api_key,
        )

    sys.exit(f"Unknown PROVIDER={PROVIDER!r}. Use: ollama | google | opencode")


def build_tools():
    """Search backend.

    Note: Gemini's built-in `google_search` grounding is NOT available on the
    free tier (returns 429 QUOTA), so DuckDuckGo is the default.
    """
    if SEARCH == "google":
        return [{"google_search": {}}]
    return [internet_search]


def ping_model() -> None:
    """Send one tiny request to check the model works, no tools involved."""
    model = build_model()
    if PROVIDER == "ollama":
        print(f"Pinging local model '{OLLAMA_MODEL}' via {OLLAMA_BASE_URL} ...")
        print("(first call loads the model into memory — this can take a while)")
    else:
        print(f"Pinging '{PROVIDER}' ...")
    reply = model.invoke("Reply with exactly one word: pong")
    print("Response:", _text_of(reply.content))


def build_agent(model_override: str | None = None):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=build_model(model_override),
        tools=build_tools(),
        system_prompt=research_instructions,
    )


def _model_candidates() -> list[str | None]:
    """Models to try in order. Only Gemini has useful fallbacks."""
    if PROVIDER != "google":
        return [None]
    ordered = [GOOGLE_MODEL] + [m for m in GOOGLE_FALLBACKS if m != GOOGLE_MODEL]
    return ordered


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def run_with_retry(query: str, on_event=None):
    """Run the agent, riding out transient model congestion.

    Free-tier Gemini returns 503 ("high demand") in waves that outlast the SDK's
    own retries, so retry the run and fall through to backup models. "start" is
    emitted once, and each fresh attempt emits "reset" so consumers can replace
    the previous attempt's trace instead of showing it twice.
    """
    emit = on_event or _print_event
    candidates = _model_candidates()
    last_exc = None

    emit({"t": "start", "query": query})

    for model_index, model_name in enumerate(candidates):
        label = model_name or PROVIDER
        for attempt in range(1, RETRIES_PER_MODEL + 1):
            if model_index or attempt > 1:
                emit({
                    "t": "reset",
                    "model": label,
                    "attempt": attempt,
                    "note": (
                        f"falling back to {label}"
                        if model_index
                        else f"retrying {label} (attempt {attempt}/{RETRIES_PER_MODEL})"
                    ),
                })
            try:
                agent = build_agent(model_name)
                return run_agent(agent, query, on_event=on_event)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_transient(exc):
                    emit({"t": "error", "message": str(exc)})
                    raise
                if attempt < RETRIES_PER_MODEL:
                    wait = 5 * attempt
                    emit({
                        "t": "retry",
                        "model": label,
                        "reason": str(exc).splitlines()[0][:100],
                        "attempt": attempt,
                        "attempts": RETRIES_PER_MODEL,
                        "wait": wait,
                    })
                    time.sleep(wait)

    emit({"t": "error", "message": str(last_exc)})
    raise last_exc  # type: ignore[misc]


SAMPLE_QUERY = "What is langgraph?"


def _preview(value, limit: int = 100) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _text_of(content) -> str:
    """Normalise message content.

    langchain-openai returns a plain string; langchain-google-genai returns a
    list of content blocks, so handle both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_agent(agent, query: str, on_event=None):
    """Stream the agent's steps as events. Returns the final answer.

    Events keep the run observable from both the CLI and the web UI.
    """
    emit = on_event or _print_event
    from langchain_core.messages import AIMessage, ToolMessage

    calls = 0
    answer = None

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            for message in update.get("messages") or []:
                if isinstance(message, AIMessage):
                    for call in message.tool_calls or []:
                        calls += 1
                        emit({
                            "t": "tool_call",
                            "node": node,
                            "name": call["name"],
                            "args": call.get("args", {}),
                        })
                    text = _text_of(message.content).strip()
                    if text:
                        answer = text
                elif isinstance(message, ToolMessage):
                    text = str(message.content)
                    if isinstance(getattr(message, "artifact", None), list):
                        text = json.dumps(message.artifact, default=str)
                    emit({
                        "t": "tool_result",
                        "node": node,
                        "name": message.name,
                        "chars": len(text),
                        "preview": text[:110].replace("\n", " ") + "…",
                    })

    emit({"t": "done", "calls": calls})
    return answer


def _print_event(event: dict) -> None:
    """Render an event for the terminal."""
    kind = event.get("t")
    if kind == "start":
        print(f"Asking: {event['query']}\n")
    elif kind == "tool_call":
        print(f"  [{event['node']}] -> {event['name']}({_preview(event.get('args', {}))})")
    elif kind == "tool_result":
        print(f"  [{event['node']}] <- {event['name']} returned {event['chars']} chars")
        print(f"           {event['preview']}")
    elif kind == "reset":
        print(f"\n--- {event['note']} ---")
    elif kind == "retry":
        print(
            f"\n[{event['model']}: {event['reason']} — retrying in "
            f"{event['wait']}s (attempt {event['attempt']}/{event['attempts']})]"
        )
    elif kind == "done":
        calls = event["calls"]
        label = "no tool calls made" if not calls else f"{calls} tool call(s) made"
        print(f"\n[{label}]")
    elif kind == "saved":
        print(f"\n[saved: {event['path']}]")
    elif kind == "error":
        print(f"\n[error: {event['message']}]", file=sys.stderr)


def _json_emitter(event: dict) -> None:
    """Write an event as one JSON line (NDJSON) so the web UI can consume it."""
    print(json.dumps(event, default=str), flush=True)


def _slug(text: str, limit: int = 60) -> str:
    """Turn a question into a safe, readable filename fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "report"


def save_report(query: str, answer: str) -> Path | None:
    """Write the report to REPORTS_DIR as markdown. Returns the path, or None."""
    if not answer:
        return None
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d-%H%M%S")
        path = REPORTS_DIR / f"{stamp}-{_slug(query)}.md"
        if PROVIDER == "google":
            model = GOOGLE_MODEL
        elif PROVIDER == "ollama":
            model = OLLAMA_MODEL
        else:
            model = OPENCODE_MODEL
        path.write_text(
            f"# {query}\n\n"
            f"*{time.strftime('%Y-%m-%d %H:%M')} · {PROVIDER}/{model}*\n\n"
            f"---\n\n{answer}\n",
            encoding="utf-8",
        )
        return path
    except Exception as exc:  # noqa: BLE001 - never fail a run over a save
        print(f"[could not save report: {exc}]", file=sys.stderr)
        return None


USAGE = """\
agents — a research agent that runs on your machine.

  agents                       open the app in your browser (the default)
  agents "your question"       research something in the terminal
  agents --ui                  open the app in your browser
  agents --check               verify setup without calling the model
  agents --ping                test the model with one tiny call
  agents --help                this message

Advanced:
  agents --events-json "..."   stream run events as JSON (the app uses this)

Runs on a free Gemini key. Search tries Tavily when TAVILY_API_KEY is set,
then DuckDuckGo/Bing. Config lives in ~/deepagents-quickstart/.env
Reports are saved to ~/research-reports/ (override with REPORTS_DIR).
"""


def main() -> None:
    args = list(sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(USAGE)
        return

    # Bare `agents`, or `agents --ui`, drops you into the app itself — the way
    # `opencode` opens its own environment when called with no arguments.
    if not args or "--ui" in args:
        print(USAGE)
        try:
            import webapp
        except ImportError:
            sys.exit(
                "The app needs its extras:\n  uv add fastapi uvicorn\n"
                'Or just ask directly:  agents "your question"'
            )
        webapp.serve()
        return

    if "--ping" in args:
        ping_model()
        return

    if "--check" in args:
        build_agent()
        print("Agent built successfully (no model calls made).")
        print(f"  provider : {PROVIDER}")
        if PROVIDER == "ollama":
            print(f"  model    : {OLLAMA_MODEL} (local, free)")
            print(f"  endpoint : {OLLAMA_BASE_URL}")
        elif PROVIDER == "google":
            print(f"  model    : {GOOGLE_MODEL} (free tier)")
            print(f"  fallbacks: {', '.join(GOOGLE_FALLBACKS)}")
        else:
            print(f"  model    : {OPENCODE_MODEL}")
            print(f"  endpoint : {OPENCODE_BASE_URL}")
        print("  built-in : file system (ls/read_file/write_file/edit_file/glob/grep)")
        if SEARCH == "google":
            search_label = "google_search (built-in)"
        else:
            search_label = f"internet_search ({', '.join(_search_sources())})"
        print(f"  search   : {search_label}")
        print(f"  reports  : {REPORTS_DIR}")
        return

    json_mode = "--events-json" in args
    positional = [a for a in args if not a.startswith("-")]
    query = " ".join(positional) if positional else SAMPLE_QUERY
    emit = _json_emitter if json_mode else None

    try:
        answer = run_with_retry(query, on_event=emit)
    except Exception as exc:  # noqa: BLE001
        if not json_mode:
            raise
        _json_emitter({"t": "error", "message": str(exc)})
        sys.exit(1)

    if not answer:
        if json_mode:
            _json_emitter({"t": "error", "message": "no final answer was produced"})
        else:
            print("(the agent finished without a final answer — see the trace above)")
        return

    if json_mode:
        _json_emitter({"t": "answer", "text": answer})
        saved = save_report(query, answer)
        if saved:
            _json_emitter({"t": "saved", "path": str(saved)})
        return

    print("\n" + "=" * 70 + "\n")
    print(answer)

    saved = save_report(query, answer)
    if saved:
        print(f"\n[saved: {saved}]")


if __name__ == "__main__":
    main()
