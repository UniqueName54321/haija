"""A basic Tkinter GUI: load a project, run the game live, and export a .txt.

Stdlib-only (``tkinter``). Launched via ``haija gui``.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, ttk
import tkinter as tk

from .config import ProjectConfig
from .engine import Engine, run_game
from .framework import load_framework
from .provider import ChatProvider, ProviderError


def format_event(ev: dict) -> str:
    t = ev.get("type")
    if t == "game_start":
        out = f"=== {ev['name']} ===\n"
        if ev.get("description"):
            out += ev["description"] + "\n"
        out += (
            f"Objective: {ev.get('objective') or '(none)'} · "
            f"Agents: {', '.join(ev.get('agents', []))}\n"
            f"Order: {ev.get('order')} · max_turns: {ev.get('max_turns')}"
        )
        return out
    if t == "turn_start":
        return f"\n--- Turn {ev['turn']} — {ev['agent']} ---"
    if t == "assistant":
        return f"[{ev['agent']} says] {ev['content']}"
    if t == "reasoning":
        return f"[{ev['agent']} thinks] {ev['reasoning']}"
    if t == "tool":
        return f"[{ev['agent']} → {ev['name']}({json.dumps(ev.get('arguments') or {})})]"
    if t == "message":
        return f"[{ev['from']} → {ev['to']}] {ev['text']}"
    if t == "log":
        return f"[{ev['from']} logs] {ev['text']}"
    if t == "outcome":
        return (
            f"[{ev['from']} declares] {ev['outcome']} "
            f"(winner {ev.get('winner') or '?'}): {ev.get('reason', '')}"
        )
    if t == "game_over":
        out = "\n=== GAME OVER ==="
        if ev.get("outcome"):
            out += f"\nOutcome: {ev['outcome']} (winner {ev.get('winner') or '?'})"
        else:
            out += "\nTurn limit reached — no outcome declared."
        out += "\nFinal state:\n" + json.dumps(ev.get("final_state", {}), indent=2)
        return out
    return json.dumps(ev)


class HaijaApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Haija — AI game engine")
        root.geometry("920x660")
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.cfg: ProjectConfig | None = None
        self.fw = None
        self.engine: Engine | None = None
        self.worker: threading.Thread | None = None
        self._build()
        root.after(100, self._poll)

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Project:").pack(side="left")
        self.path_var = tk.StringVar(value=".")
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Browse", command=self._browse).pack(side="left")
        self.run_btn = ttk.Button(top, text="Run", command=self._run)
        self.run_btn.pack(side="left", padx=6)
        self.export_btn = ttk.Button(top, text="Export .txt", command=self._export, state="disabled")
        self.export_btn.pack(side="left")

        mid = ttk.Frame(self.root, padding=8)
        mid.pack(fill="both", expand=True)
        self.log = tk.Text(mid, wrap="word", state="disabled", font=("Consolas", 10))
        sb = ttk.Scrollbar(mid, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, padding=4, relief="sunken").pack(fill="x")

    def _append(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _browse(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.path_var.set(d)

    def _load(self) -> Path:
        p = Path(self.path_var.get() or ".")
        toml = p / "haija.toml" if p.is_dir() else p
        if not toml.exists():
            raise FileNotFoundError(f"no haija.toml found at {p}")
        self.cfg = ProjectConfig.load(toml)
        self.fw = load_framework(toml.parent / self.cfg.framework_path)
        return toml.parent

    # ---- run --------------------------------------------------------------
    def _run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            base = self._load()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load error", str(e))
            return
        self._clear_log()
        self.run_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.status.set("Running…")

        def observer(event: dict) -> None:
            self.q.put(event)

        self.worker = threading.Thread(
            target=self._run_thread, args=(base, observer), daemon=True
        )
        self.worker.start()

    def _run_thread(self, base: Path, observer) -> None:
        try:
            provider = ChatProvider(self.cfg.model)
            engine = Engine(
                self.fw,
                [a.name for a in self.cfg.agents],
                max_steps_per_turn=self.cfg.max_steps_per_turn,
            )
            self.engine = engine
            persist_msg = run_game(engine, provider, self.cfg, observer=observer)
            self.q.put({"type": "info", "message": persist_msg})
        except ProviderError as e:
            self.q.put({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            self.q.put({"type": "error", "message": str(e)})
        finally:
            self.q.put({"type": "done"})

    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "error":
            self._append(f"[error] {ev['message']}")
            self.status.set("Error")
            self.run_btn.configure(state="normal")
        elif t == "done":
            self.status.set("Finished — click Export .txt to save the run")
            self.run_btn.configure(state="normal")
            self.export_btn.configure(state="normal")
        elif t == "info":
            self.status.set(ev["message"])
        else:
            self._append(format_event(ev))

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ---- export -----------------------------------------------------------
    def _export(self) -> None:
        if not self.engine:
            messagebox.showinfo("Nothing to export", "Run a game first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="haija-export.txt"
        )
        if not path:
            return
        Path(path).write_text(self.engine.export_text(), encoding="utf-8")
        self.status.set(f"Exported → {path}")


def main() -> int:
    root = Tk()
    HaijaApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
