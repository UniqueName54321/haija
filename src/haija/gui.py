"""Haija's graphical interface — a zero-dependency browser app.

``haija gui`` starts a local HTTP server (stdlib ``http.server``) and opens your
browser. The UI has full feature parity with the CLI: open/new a project,
generate a framework from a prompt, validate, run (with live streaming), tweak
options (tone / archetypes / model), and export the run as a ``.txt``.

No Tkinter, no platform-specific quirks — just Python + a browser.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import ProjectConfig
from .engine import Engine, run_game
from .framework import load_framework
from .generate import generate_framework
from .project import new_project, projects_root
from .provider import ChatProvider, ProviderError

DEFAULT_PORT = 8657


@dataclass
class HaijaState:
    project_dir: Path = field(default_factory=lambda: Path("."))
    cfg: ProjectConfig | None = None
    fw: Any = None
    toml_path: Path | None = None
    engine: Engine | None = None
    export_path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    busy: bool = False
    worker_thread: threading.Thread | None = None

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _read_index_html() -> str:
    try:
        from importlib import resources

        return resources.files("haija").joinpath("static", "index.html").read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return (
            "<!doctype html><html><body style='font-family:sans-serif;background:#0f1220;"
            "color:#dbe0ee;padding:2rem'><h1>Haija</h1><p>The UI failed to load.</p>"
            "<p>Make sure the <code>haija/static/index.html</code> file is present.</p></body></html>"
        )


def _run_thread(state: HaijaState) -> None:
    try:
        provider = ChatProvider(state.cfg.model)
        engine = Engine(
            state.fw,
            [a.name for a in state.cfg.agents],
            max_steps_per_turn=state.cfg.max_steps_per_turn,
        )
        state.engine = engine
        persist_msg = run_game(engine, provider, state.cfg, observer=state.add_event)
        state.add_event({"type": "info", "message": persist_msg})
    except ProviderError as e:
        state.add_event({"type": "error", "message": str(e)})
    except Exception as e:  # noqa: BLE001
        state.add_event({"type": "error", "message": str(e)})
    finally:
        state.add_event({"type": "done"})
        state.busy = False


def _generate_thread(state: HaijaState, prompt: str) -> None:
    try:
        provider = ChatProvider(state.cfg.model)
        fw = generate_framework(provider, prompt, name=state.cfg.name)
        out = state.toml_path.parent / state.cfg.framework_path
        out.write_text(json.dumps(fw.to_dict(), indent=2) + "\n", encoding="utf-8")
        state.fw = fw
        state.add_event({"type": "info", "message": f"Wrote framework '{fw.name}' → {out}"})
    except Exception as e:  # noqa: BLE001
        state.add_event({"type": "error", "message": str(e)})
    finally:
        state.add_event({"type": "done"})
        state.busy = False


class HaijaHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler):
        super().__init__(addr, handler)
        self.state = HaijaState()


class Handler(BaseHTTPRequestHandler):
    server: HaijaHTTPServer

    def log_message(self, *args) -> None:  # silence request logging
        pass

    @property
    def state(self) -> HaijaState:
        return self.server.state

    # ---- helpers ----------------------------------------------------------
    def _send_json(self, obj: Any, status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _query(self, name: str, default: str = "") -> str:
        qs = parse_qs(urlparse(self.path).query)
        return qs.get(name, [default])[0]

    # ---- GET --------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/events":
            self._api_events()
        elif path == "/api/replay":
            self._api_replay()
        elif path == "/api/download":
            self._api_download()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        data = _read_index_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _api_status(self) -> None:
        s = self.state
        if not s.cfg:
            self._send_json({"project": None, "agents": [], "archetypes": [], "busy": s.busy})
            return
        archetypes = [
            {
                "id": a.id,
                "name": a.name,
                "persona": a.persona,
                "enabled": s.cfg.agent_for_archetype(a.id) is not None,
                "custom": a.id in s.cfg.archetypes,
            }
            for a in sorted(s.cfg.all_archetypes().values(), key=lambda x: x.id)
        ]
        self._send_json(
            {
                "project": s.cfg.name,
                "dir": str(s.project_dir),
                "tone": s.cfg.tone,
                "model_provider": s.cfg.model.provider,
                "model_model": s.cfg.model.model,
                "model_base_url": s.cfg.model.base_url,
                "model_api_key_env": s.cfg.model.api_key_env,
                "agents": [
                    {
                        "name": a.name,
                        "archetype": a.archetype,
                        "description": a.description,
                        "thinking": a.thinking,
                    }
                    for a in s.cfg.agents
                ],
                "archetypes": archetypes,
                "busy": s.busy,
            }
        )

    def _api_events(self) -> None:
        try:
            since = int(self._query("since", "0"))
        except ValueError:
            since = 0
        events = self.state.events[since:]
        self._send_json({"events": events, "next": len(self.state.events)})

    def _api_replay(self) -> None:
        s = self.state
        if not s.cfg:
            self._send_json({"ok": False, "message": "open a project first"})
            return
        base = s.toml_path.parent
        run_path = base / "run.json"
        if not run_path.exists():
            self._send_json({"ok": False, "message": "no run yet — run a game first"})
            return
        try:
            run_log = json.loads(run_path.read_text(encoding="utf-8"))
            fw = load_framework(base / s.cfg.framework_path)
            state_path = base / s.cfg.state_path
            final_state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.exists()
                else {}
            )
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "message": str(e)})
            return
        self._send_json(
            {
                "ok": True,
                "name": fw.name,
                "description": fw.description,
                "objective": fw.objective,
                "agents": [a.name for a in s.cfg.agents],
                "max_turns": fw.turn.max_turns,
                "run_log": run_log,
                "final_state": final_state,
            }
        )

    def _api_download(self) -> None:
        p = self.state.export_path
        if not p or not p.exists():
            self.send_error(404)
            return
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- POST -------------------------------------------------------------
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_body()
        routes = {
            "/api/open": self._api_open,
            "/api/new": self._api_new,
            "/api/generate": self._api_generate,
            "/api/validate": self._api_validate,
            "/api/run": self._api_run,
            "/api/export": self._api_export,
            "/api/options": self._api_options,
        }
        handler = routes.get(path)
        if handler:
            handler(body)
        else:
            self.send_error(404)

    def _api_open(self, body: dict) -> None:
        raw = (body.get("path") or "").strip() or "."
        d = Path(raw)
        toml = d / "haija.toml" if d.is_dir() else d
        if not toml.exists():
            self._send_json({"ok": False, "message": f"no haija.toml found at {raw}"})
            return
        try:
            cfg = ProjectConfig.load(toml)
            fw = load_framework(toml.parent / cfg.framework_path)
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "message": str(e)})
            return
        s = self.state
        s.cfg, s.toml_path, s.fw, s.project_dir = cfg, toml, fw, toml.parent
        self._send_json({"ok": True, "message": f"Loaded project '{cfg.name}'", "project": cfg.name})

    def _api_new(self, body: dict) -> None:
        name = (body.get("name") or "").strip()
        directory = (body.get("dir") or "").strip()
        if not name:
            self._send_json({"ok": False, "message": "project name is required"})
            return
        dest = Path(directory) if directory else projects_root()
        try:
            root = new_project(name, dest)
        except FileExistsError:
            self._send_json({"ok": False, "message": f"'{name}' already exists"})
            return
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "message": str(e)})
            return
        cfg = ProjectConfig.load(root / "haija.toml")
        fw = load_framework(root / cfg.framework_path)
        s = self.state
        s.cfg, s.toml_path, s.fw, s.project_dir = cfg, root / "haija.toml", fw, root
        self._send_json({"ok": True, "message": f"Created project '{name}' at {root}", "project": name})

    def _api_generate(self, body: dict) -> None:
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._send_json({"ok": False, "message": "prompt is required"})
            return
        s = self.state
        if not s.cfg:
            self._send_json({"ok": False, "message": "open a project first"})
            return
        if s.busy:
            self._send_json({"ok": False, "message": "a task is already running"})
            return
        s.busy = True
        s.events.clear()
        s.worker_thread = threading.Thread(target=_generate_thread, args=(s, prompt), daemon=True)
        s.worker_thread.start()
        self._send_json({"ok": True, "message": "generating…"})

    def _api_validate(self, body: dict) -> None:
        s = self.state
        if not s.cfg:
            self._send_json({"ok": False, "message": "open a project first"})
            return
        try:
            s.fw = load_framework(s.toml_path.parent / s.cfg.framework_path)
            self._send_json(
                {
                    "ok": True,
                    "message": f"OK — framework '{s.fw.name}' "
                    f"({len(s.fw.actions)} actions, {len(s.cfg.agents)} agents)",
                }
            )
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "message": str(e)})

    def _api_run(self, body: dict) -> None:
        s = self.state
        if not s.cfg:
            self._send_json({"ok": False, "message": "open a project first"})
            return
        if s.busy:
            self._send_json({"ok": False, "message": "a task is already running"})
            return
        s.busy = True
        s.events.clear()
        s.worker_thread = threading.Thread(target=_run_thread, args=(s,), daemon=True)
        s.worker_thread.start()
        self._send_json({"ok": True, "message": "running…"})

    def _api_export(self, body: dict) -> None:
        s = self.state
        if not s.engine:
            self._send_json({"ok": False, "message": "nothing to export — run a game first"})
            return
        if body.get("path"):
            out = Path(body["path"])
        else:
            out = s.project_dir / "haija-export.txt"
        try:
            out.write_text(s.engine.export_text(), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "message": str(e)})
            return
        s.export_path = out
        self._send_json(
            {"ok": True, "message": f"Exported → {out}", "path": str(out), "download": "/api/download"}
        )

    def _api_options(self, body: dict) -> None:
        s = self.state
        if not s.cfg:
            self._send_json({"ok": False, "message": "open a project first"})
            return
        cfg = s.cfg
        if "tone" in body and body["tone"] is not None:
            cfg.tone = (body["tone"] or "").strip()
        if body.get("model"):
            cfg.model.model = body["model"].strip()
        if body.get("base_url"):
            cfg.model.base_url = body["base_url"].strip()
        if body.get("api_key_env"):
            cfg.model.api_key_env = body["api_key_env"].strip()
        if body.get("enabled_archetypes") is not None:
            enabled = set(body["enabled_archetypes"])
            for arch_id in list(cfg.all_archetypes()):
                cfg.set_archetype_enabled(arch_id, arch_id in enabled)
        if body.get("agent_thinking"):
            for name, level in body["agent_thinking"].items():
                lvl = str(level or "").strip().lower()
                cfg.set_agent_thinking(name, None if lvl in ("", "default") else lvl)
        try:
            cfg.save(s.toml_path)
        except Exception as e:  # noqa: BLE001
            self._send_json({"ok": False, "message": str(e)})
            return
        self._send_json({"ok": True, "message": f"Options saved → {s.toml_path}"})


def main() -> int:
    host = "127.0.0.1"
    port = int(os.environ.get("HAIJA_PORT", DEFAULT_PORT))
    server = HaijaHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Haija GUI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
