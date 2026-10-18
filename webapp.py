"""Local web UI for the research agent.

Run with:  agents --ui

The UI is a thin shell over the CLI. Each question spawns
`research_agent.py --events-json` as a subprocess and forwards its NDJSON
events to the browser over SSE. That keeps one source of truth for the agent
logic, guarantees every run picks up the current .env, and avoids reloading
module state inside a long-lived server.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

PROJECT = Path(__file__).resolve().parent
ENV_PATH = PROJECT / ".env"
ENV_EXAMPLE = PROJECT / ".env.example"
STATIC = PROJECT / "static"

load_dotenv(ENV_PATH)
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "~/research-reports")).expanduser()

# Models offered in the UI. Free-tier Gemini flash models plus whatever the
# local Ollama install has, if any.
GOOGLE_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

app = FastAPI(title="local-agents")


# --- .env handling ------------------------------------------------------------
# The .env holds secrets that cannot be regenerated, so edits are surgical:
# only the named keys are touched, every other line (including the API key) is
# preserved byte for byte.


def _env_lines() -> list[str]:
    if ENV_PATH.exists():
        return ENV_PATH.read_text(encoding="utf-8").splitlines()
    if ENV_EXAMPLE.exists():
        return ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    return []


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _env_lines():
        match = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def update_env(updates: dict[str, str]) -> None:
    """Set the given keys, leaving every other line untouched."""
    if not updates:
        return
    remaining = dict(updates)
    out: list[str] = []
    for line in _env_lines():
        match = re.match(r"^\s*#?\s*([A-Z0-9_]+)\s*=", line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def _mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 10:
        return "•" * len(secret)
    return f"{secret[:6]}…{secret[-4:]}"


def config_payload() -> dict:
    env = read_env()
    provider = (env.get("PROVIDER") or "google").lower()
    sources = [b for b in (env.get("SEARCH_BACKENDS") or "duckduckgo,bing").split(",") if b]
    if env.get("TAVILY_API_KEY"):
        sources.insert(0, "tavily")

    models = list(GOOGLE_MODELS)
    if provider == "ollama":
        models = _ollama_models() or ["gemma4:26b", "gemma4:latest"]
    current = env.get("GOOGLE_MODEL") if provider == "google" else env.get("OLLAMA_MODEL")
    if current and current not in models:
        models.insert(0, current)

    return {
        "provider": provider,
        "model": current or (models[0] if models else ""),
        "models": models,
        "providers": ["google", "ollama"],
        "fallbacks": [m for m in (env.get("GOOGLE_FALLBACKS") or "").split(",") if m],
        "search_sources": sources,
        "google_key": _mask(env.get("GOOGLE_API_KEY", "")),
        "tavily_key": _mask(env.get("TAVILY_API_KEY", "")),
        "reports_dir": str(REPORTS_DIR),
    }


def _ollama_models() -> list[str]:
    if not shutil.which("ollama"):
        return []
    try:
        import subprocess

        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        ).stdout
        names = []
        for line in out.splitlines()[1:]:
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names
    except Exception:  # noqa: BLE001 - a missing ollama must not break the UI
        return []


# --- config API ---------------------------------------------------------------


@app.get("/api/config")
def get_config():
    return config_payload()


@app.post("/api/config")
async def post_config(payload: dict):
    allowed = {
        "GOOGLE_MODEL",
        "OLLAMA_MODEL",
        "PROVIDER",
        "GOOGLE_FALLBACKS",
        "TAVILY_API_KEY",
        "GOOGLE_API_KEY",
        "SEARCH_BACKENDS",
    }
    updates: dict[str, str] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        value = ("" if value is None else str(value)).strip()
        # An empty secret means "leave it alone", so a blank form field can never
        # wipe a key the user cannot retype from memory.
        if key.endswith("_KEY") and not value:
            continue
        updates[key] = value

    update_env(updates)
    return config_payload()


# --- reports ------------------------------------------------------------------


@app.get("/api/reports")
def list_reports():
    if not REPORTS_DIR.exists():
        return []
    items = []
    for path in sorted(REPORTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        title = path.stem
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            if first.startswith("# "):
                title = first[2:].strip()
        except Exception:  # noqa: BLE001
            pass
        items.append({
            "name": path.name,
            "title": title,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return items


def _safe_report_path(name: str) -> Path:
    # Reject anything that tries to escape the reports directory.
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid report name")
    path = (REPORTS_DIR / name).resolve()
    if path.parent != REPORTS_DIR.resolve() or not path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return path


@app.get("/api/reports/{name}")
def get_report(name: str):
    path = _safe_report_path(name)
    return {"name": path.name, "content": path.read_text(encoding="utf-8")}


@app.delete("/api/reports/{name}")
def delete_report(name: str):
    path = _safe_report_path(name)
    path.unlink()
    return {"deleted": path.name}


# --- running the agent --------------------------------------------------------


@app.get("/api/ask")
async def ask(q: str):
    """Stream a run as Server-Sent Events."""
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty question")

    async def event_stream():
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(PROJECT / "research_agent.py"),
            "--events-json",
            q,
            cwd=str(PROJECT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                # Only forward lines that are valid event objects; stray output
                # (warnings, tracebacks) is surfaced as a log line.
                try:
                    json.loads(line)
                except ValueError:
                    line = json.dumps({"t": "log", "message": line})
                yield f"data: {line}\n\n"
            await proc.wait()
        finally:
            if proc.returncode is None:
                proc.kill()
        yield 'data: {"t":"end"}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}"
    print(f"local-agents UI → {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    serve()
