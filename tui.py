"""The local-agents terminal app.

Run with:  agents

Same agent as the CLI and the same event stream the browser UI uses, but it
lives in the terminal: ask a question, watch the run, read the report, and open
anything from the archive without leaving the window.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

import settings

PROJECT = settings.PROJECT

WELCOME = """\
# local-agents

A research agent that runs on your machine. Ask a question below, watch it
search, then read the report here or open it from the archive on the left.

## In here

- `ctrl+n` focus the question box
- `enter` start a run
- `escape` stop a run that is going
- `ctrl+o` open the report in `$EDITOR`
- `ctrl+r` refresh the archive
- `ctrl+s` settings — provider, model, keys
- `ctrl+q` quit

Type straight into the box at the top. Plain letters go to your question, which
is why the shortcuts carry `ctrl`.

## From the shell

- `agents` this app
- `agents "your question"` research straight from the command line
- `agents --browser` the browser version, easier for reading long reports
- `agents --check` verify setup, no model calls
- `agents --ping` test the model with one tiny call
- `agents --help` full usage
"""

CSS = """
Screen {
    background: #14161a;
    color: #e8e6e1;
}

#titlebar {
    dock: top;
    height: 1;
    background: #1c1f26;
    padding: 0 1;
}
#brand { width: auto; color: #e8e6e1; text-style: bold; }
#status { width: 1fr; color: #6e7480; padding: 0 2; }
#model { width: auto; color: #6e7480; }

#body { height: 1fr; }

#left {
    width: 34;
    background: #171a20;
    border-right: solid #2a2f38;
}
#left-head {
    height: 1;
    color: #6e7480;
    padding: 0 1;
}
#reports { height: 1fr; background: transparent; }
#reports > ListItem { height: auto; padding: 0 1; }
#reports > ListItem.-highlight { background: #1f4fd8 35%; }
.r-title { width: 100%; color: #e8e6e1; }
.r-meta { color: #6e7480; }
.r-empty { color: #6e7480; padding: 1; }

#right { width: 1fr; }

#question {
    dock: top;
    margin: 0;
    border: tall #2a2f38;
    background: #1c1f26;
    color: #e8e6e1;
}
#question:focus { border: tall #6b8cff; }

#tabs { height: 1fr; }
#report-scroll { height: 1fr; padding: 1 2; }
#log { height: 1fr; background: transparent; padding: 0 1; }

Footer { background: #1c1f26; }

#sheet {
    width: 66;
    height: auto;
    max-height: 90%;
    background: #1c1f26;
    border: tall #2a2f38;
    padding: 1 2;
}
#sheet-title { text-style: bold; color: #e8e6e1; }
.sheet-note { color: #6e7480; margin-bottom: 1; }
.field-label { color: #9aa0ab; margin-top: 1; }
#sheet-buttons { height: auto; align-horizontal: right; margin-top: 2; }
#sheet-buttons Button { margin-left: 2; }
"""


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%b %d %H:%M")


def _read_reports() -> list[tuple[str, str, float]]:
    """(filename, title, mtime) for every saved report, newest first."""
    reports_dir = settings.REPORTS_DIR
    if not reports_dir.exists():
        return []
    found = []
    for path in reports_dir.glob("*.md"):
        try:
            stat = path.stat()
        except OSError:
            continue
        title = path.stem
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            if first.startswith("# "):
                title = first[2:].strip()
        except Exception:  # noqa: BLE001 - a title is never worth crashing for
            pass
        found.append((path.name, title, stat.st_mtime))
    found.sort(key=lambda item: item[2], reverse=True)
    return found


class SettingsScreen(ModalScreen):
    """Edit the same .env the CLI reads."""

    BINDINGS = [Binding("escape", "dismiss", "cancel")]

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        cfg = self.config
        with Vertical(id="sheet"):
            yield Static("Settings", id="sheet-title")
            yield Static("Saved to .env. The next run picks it up.", classes="sheet-note")

            yield Label("Provider", classes="field-label")
            yield Select(
                [(p, p) for p in cfg["providers"]],
                value=cfg["provider"],
                allow_blank=False,
                id="s-provider",
            )

            yield Label("Model", classes="field-label")
            yield Select(
                [(m, m) for m in cfg["models"]],
                value=cfg["model"],
                allow_blank=False,
                id="s-model",
            )

            yield Label("Fallback models", classes="field-label")
            yield Input(value=", ".join(cfg["fallbacks"]), id="s-fallbacks")

            yield Label("Tavily API key", classes="field-label")
            yield Input(
                password=True,
                placeholder=(
                    f"currently {cfg['tavily_key']}" if cfg["tavily_key"] else "not set"
                ),
                id="s-tavily",
            )

            yield Label("Google API key", classes="field-label")
            yield Input(
                password=True,
                placeholder=(
                    f"currently {cfg['google_key']}" if cfg["google_key"] else "not set"
                ),
                id="s-google",
            )

            with Horizontal(id="sheet-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="save")

    @on(Select.Changed, "#s-provider")
    def provider_changed(self, event: Select.Changed) -> None:
        if event.value == "ollama":
            models = settings.ollama_models() or ["gemma4:26b", "gemma4:latest"]
        else:
            models = list(settings.GOOGLE_MODELS)
        current = self.query_one("#s-model", Select)
        current.set_options([(m, m) for m in models])

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    def save(self) -> None:
        provider = self.query_one("#s-provider", Select).value
        payload = {
            "PROVIDER": str(provider),
            "GOOGLE_FALLBACKS": self.query_one("#s-fallbacks", Input).value.strip(),
        }
        model = self.query_one("#s-model", Select).value
        if model and model is not Select.BLANK:
            key = "OLLAMA_MODEL" if provider == "ollama" else "GOOGLE_MODEL"
            payload[key] = str(model)

        # A blank secret field means "leave it alone".
        for field, env_key in (("#s-tavily", "TAVILY_API_KEY"), ("#s-google", "GOOGLE_API_KEY")):
            value = self.query_one(field, Input).value.strip()
            if value:
                payload[env_key] = value

        self.dismiss(payload)


class Agents(App):
    """The terminal front-end."""

    CSS = CSS
    TITLE = "local-agents"
    BINDINGS = [
        Binding("escape", "cancel_run", "stop"),
        Binding("ctrl+n", "focus_question", "ask"),
        Binding("ctrl+o", "open_external", "open"),
        Binding("ctrl+r", "refresh", "refresh"),
        Binding("ctrl+s", "settings", "settings"),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = settings.config_payload()
        self.reports: list[tuple[str, str, float]] = []
        self.current: str | None = None
        self.step = 0
        self.running = False
        self.cancelled = False
        self.proc: subprocess.Popen | None = None

    # --- layout ---------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="titlebar"):
            yield Static("local-agents", id="brand")
            yield Static("", id="status")
            yield Static(self._model_label(), id="model")

        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("Reports", id="left-head")
                yield ListView(id="reports")
            with Vertical(id="right"):
                yield Input(
                    placeholder="Ask a research question…",
                    id="question",
                )
                with TabbedContent(initial="pane-report", id="tabs"):
                    with TabPane("Report", id="pane-report"):
                        with VerticalScroll(id="report-scroll"):
                            yield Markdown(WELCOME, id="report")
                    with TabPane("Run log", id="pane-log"):
                        yield RichLog(id="log", highlight=False, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.load_reports()
        self.query_one("#question", Input).focus()

    def _model_label(self) -> str:
        cfg = self.config
        return f"{cfg['provider']}/{cfg['model']}"

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # --- reports --------------------------------------------------------------

    def load_reports(self) -> None:
        self.reports = _read_reports()
        listing = self.query_one("#reports", ListView)
        listing.clear()
        if not self.reports:
            listing.append(ListItem(Label("Nothing yet.", classes="r-empty")))
            return
        for name, title, mtime in self.reports:
            label = title if len(title) <= 40 else title[:39] + "…"
            listing.append(
                ListItem(
                    Label(escape(label), classes="r-title"),
                    Label(_date(mtime), classes="r-meta"),
                )
            )

    def open_report(self, name: str) -> None:
        path = settings.REPORTS_DIR / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.notify(f"Could not read {name}: {exc}", severity="error")
            return
        self.current = name
        self.show_report(text)
        self._set_status(f"reading {name}")

    def show_report(self, text: str) -> None:
        self.query_one("#report", Markdown).update(text)
        self.query_one(TabbedContent).active = "pane-report"

    @on(ListView.Selected, "#reports")
    def report_chosen(self, event: ListView.Selected) -> None:
        if not self.reports:
            return
        index = event.index
        if index is None or index >= len(self.reports):
            return
        self.open_report(self.reports[index][0])
        self.query_one("#question", Input).focus()

    # --- running --------------------------------------------------------------

    @on(Input.Submitted, "#question")
    def submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query or self.running:
            return
        event.input.value = ""
        self.start(query)

    def start(self, query: str) -> None:
        self.running = True
        self.cancelled = False
        self.step = 0
        self.query_one("#log", RichLog).clear()
        self.query_one(TabbedContent).active = "pane-log"
        self._set_status("running…")
        self.run_question(query)

    @work(thread=True, exclusive=True, group="run")
    def run_question(self, query: str) -> None:
        """Spawn the CLI and relay its events.

        A subprocess rather than an in-process call, so every run reads the
        current .env and a wedged model call cannot take the app down with it.
        """
        try:
            self.proc = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT / "research_agent.py"),
                    "--events-json",
                    query,
                ],
                cwd=str(PROJECT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.call_from_thread(self.finish, f"could not start the agent: {exc}")
            return

        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                event = {"t": "log", "message": line}
            self.call_from_thread(self.handle, event)

        self.proc.wait()
        self.proc = None
        self.call_from_thread(self.finish, None)

    def handle(self, event: dict) -> None:
        kind = event.get("t")
        log = self.query_one("#log", RichLog)

        if kind == "start":
            self._set_status("searching…")

        elif kind == "tool_call":
            self.step += 1
            args = event.get("args") or {}
            detail = args.get("query") or json.dumps(args)
            log.write(
                f"[#6e7480]{self.step:02d}[/] [#6b8cff]→[/] "
                f"[bold]{escape(str(event.get('name', '')))}[/]  "
                f"[#9aa0ab]{escape(str(detail))}[/]"
            )

        elif kind == "tool_result":
            self.step += 1
            log.write(
                f"[#6e7480]{self.step:02d}[/] [#3fb27f]←[/] "
                f"{escape(str(event.get('name', '')))}  "
                f"[#6e7480]{event.get('chars', 0)} chars[/]"
            )
            log.write(f"     [#6e7480]{escape(str(event.get('preview', '')))}[/]")

        elif kind == "reset":
            # A retry replays the run, so drop the dead attempt's trace rather
            # than printing the same work twice.
            log.clear()
            self.step = 0
            log.write(f"[#d29922]↻ {escape(str(event.get('note', 'retrying')))}[/]")

        elif kind == "retry":
            log.write(
                f"[#d29922]… {escape(str(event.get('model', '')))} was busy — "
                f"waiting {event.get('wait', 0)}s "
                f"(attempt {event.get('attempt')}/{event.get('attempts')})[/]"
            )

        elif kind == "done":
            calls = event.get("calls", 0)
            self._set_status(
                "no tool calls" if not calls else f"{calls} tool call(s) done"
            )

        elif kind == "answer":
            self.step += 1
            self.show_report(str(event.get("text", "")))
            self._set_status("report ready")

        elif kind == "saved":
            path = Path(str(event.get("path", "")))
            self.current = path.name
            log.write(f"[#3fb27f]✓ saved to {escape(str(path))}[/]")
            self._set_status(f"saved {path.name}")

        elif kind == "error":
            log.write(f"[#e5534b]! {escape(str(event.get('message', '')))}[/]")
            self._set_status("error")

        elif kind == "log":
            log.write(f"[#6e7480]{escape(str(event.get('message', '')))}[/]")

    def finish(self, problem: str | None) -> None:
        self.running = False
        if problem:
            self.query_one("#log", RichLog).write(f"[#e5534b]! {escape(problem)}[/]")
            self._set_status("error")
        elif self.cancelled:
            self._set_status("stopped")
        self.load_reports()
        self.query_one("#question", Input).focus()

    # --- actions --------------------------------------------------------------

    def action_focus_question(self) -> None:
        self.query_one("#question", Input).focus()

    def action_cancel_run(self) -> None:
        if not self.running or not self.proc:
            return
        self.cancelled = True
        try:
            self.proc.kill()
        except OSError:
            pass

    def action_refresh(self) -> None:
        self.load_reports()
        self.notify("Archive refreshed", timeout=2)

    def action_open_external(self) -> None:
        if not self.current:
            self.notify("No report open", timeout=2)
            return
        path = settings.REPORTS_DIR / self.current
        if not path.exists():
            self.notify("That report is gone", severity="warning", timeout=3)
            return
        command = os.environ.get("EDITOR") or os.environ.get("PAGER") or "less"
        with self.suspend():
            try:
                subprocess.call([command, str(path)])
            except OSError as exc:
                print(f"could not launch {command}: {exc}")

    def action_settings(self) -> None:
        self.push_screen(SettingsScreen(settings.config_payload()), self.saved)

    def saved(self, payload: dict | None) -> None:
        if not payload:
            return
        settings.apply_updates(payload)
        self.config = settings.config_payload()
        self.query_one("#model", Static).update(self._model_label())
        self.notify("Settings saved", timeout=3)


def run() -> None:
    Agents().run()


if __name__ == "__main__":
    run()
