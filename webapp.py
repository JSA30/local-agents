"""Local web UI for the research agent.

Run with:  agents --browser

The UI is a thin shell over the CLI. Each question spawns
`research_agent.py --events-json` as a subprocess and forwards its NDJSON
events to the browser over SSE. That keeps one source of truth for the agent
logic, guarantees every run picks up the current .env, and avoids reloading
module state inside a long-lived server.
"""

from __future__ import annotations

import asyncio
import json
import sys
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

import settings

PROJECT = settings.PROJECT
STATIC = settings.STATIC

app = FastAPI(title="local-agents")


# --- config -------------------------------------------------------------------


@app.get("/api/config")
def get_config():
    return settings.config_payload()


@app.post("/api/config")
async def post_config(payload: dict):
    settings.apply_updates(payload)
    return settings.config_payload()


# --- reports ------------------------------------------------------------------


@app.get("/api/reports")
def list_reports():
    reports_dir = settings.REPORTS_DIR
    if not reports_dir.exists():
        return []
    items = []
    for path in sorted(
        reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        stat = path.stat()
        title = path.stem
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            if first.startswith("# "):
                title = first[2:].strip()
        except Exception:  # noqa: BLE001 - a title is never worth a 500
            pass
        items.append(
            {
                "name": path.name,
                "title": title,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return items


def _safe_report_path(name: str) -> Path:
    """Resolve a report name inside the reports directory, or refuse."""
    reports_dir = settings.REPORTS_DIR.resolve()
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid report name")
    path = (reports_dir / name).resolve()
    if path.parent != reports_dir or not path.exists():
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
    print(f"local-agents browser UI → {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - opening a browser is best effort
            pass
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    serve()
