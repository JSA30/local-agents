"""Shared settings: the `.env` file, and the config both front-ends display.

The terminal app and the browser app read and write the same `.env` through
here, so the surgical edit that protects an un-retypeable API key exists once.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
ENV_PATH = PROJECT / ".env"
ENV_EXAMPLE = PROJECT / ".env.example"
STATIC = PROJECT / "static"

load_dotenv(ENV_PATH)
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "~/research-reports")).expanduser()

# Models offered in the settings UI. Free-tier Gemini flash models, plus
# whatever the local Ollama install has to hand.
GOOGLE_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

PROVIDERS = ["google", "ollama"]

# Only these may be written from a UI.
EDITABLE_KEYS = {
    "PROVIDER",
    "GOOGLE_MODEL",
    "OLLAMA_MODEL",
    "GOOGLE_FALLBACKS",
    "SEARCH_BACKENDS",
    "TAVILY_API_KEY",
    "GOOGLE_API_KEY",
}


def env_lines() -> list[str]:
    if ENV_PATH.exists():
        return ENV_PATH.read_text(encoding="utf-8").splitlines()
    if ENV_EXAMPLE.exists():
        return ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    return []


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_lines():
        match = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def update_env(updates: dict[str, str]) -> None:
    """Set the given keys, leaving every other line untouched.

    The `.env` holds a key that cannot be regenerated, so this never rewrites
    the file wholesale — unmatched lines are preserved exactly as they were.
    """
    if not updates:
        return
    remaining = dict(updates)
    out: list[str] = []
    for line in env_lines():
        match = re.match(r"^\s*#?\s*([A-Z0-9_]+)\s*=", line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def apply_updates(payload: dict) -> None:
    """Filter a payload down to editable keys and write it.

    An empty secret means "leave it alone", so a blank form field can never
    wipe a key the user cannot retype from memory.
    """
    updates: dict[str, str] = {}
    for key, value in payload.items():
        if key not in EDITABLE_KEYS:
            continue
        value = ("" if value is None else str(value)).strip()
        if key.endswith("_KEY") and not value:
            continue
        updates[key] = value
    update_env(updates)


def mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 10:
        return "•" * len(secret)
    return f"{secret[:6]}…{secret[-4:]}"


def ollama_models() -> list[str]:
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:  # noqa: BLE001 - a missing ollama must not break a UI
        return []
    names = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def config_payload() -> dict:
    env = read_env()
    provider = (env.get("PROVIDER") or "google").lower()
    sources = [
        b for b in (env.get("SEARCH_BACKENDS") or "duckduckgo,bing").split(",") if b
    ]
    if env.get("TAVILY_API_KEY"):
        sources.insert(0, "tavily")

    if provider == "ollama":
        models = ollama_models() or ["gemma4:26b", "gemma4:latest"]
    else:
        models = list(GOOGLE_MODELS)
    current = (
        env.get("OLLAMA_MODEL") if provider == "ollama" else env.get("GOOGLE_MODEL")
    )
    if current and current not in models:
        models.insert(0, current)

    return {
        "provider": provider,
        "model": current or (models[0] if models else ""),
        "models": models,
        "providers": list(PROVIDERS),
        "fallbacks": [m for m in (env.get("GOOGLE_FALLBACKS") or "").split(",") if m],
        "search_sources": sources,
        "google_key": mask(env.get("GOOGLE_API_KEY", "")),
        "tavily_key": mask(env.get("TAVILY_API_KEY", "")),
        "reports_dir": str(REPORTS_DIR),
    }
