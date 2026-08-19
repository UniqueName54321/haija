"""A basic Tkinter GUI: load a project, run the game live, and export a .txt.

Also hosts the per-game **options menu**: set the game tone, enable/disable
archetypes, and define custom archetypes. Stdlib-only (``tkinter``). Launched
via ``haija gui``.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import Tk, filedialog, messagebox, ttk

from .archetypes import Archetype
from .config import ProjectConfig, dump_config
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


class NewArchetypeDialog(tk.Toplevel):
    def __init__(self, parent, on_add) -> None:
        super().__init__(parent)
        self.on_add = on_add
        self.title("New archetype")
        self.transient(parent)
        self.grab_set()
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="id (short, no spaces):").grid(row=0, column=0, sticky="w")
        self.id_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.id_var, width=40).grid(row=0, column=1, pady=3)
        ttk.Label(frm, text="Name:").grid(row=1, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var, width=40).grid(row=1, column=1, pady=3)
        ttk.Label(frm, text="Persona:").grid(row=2, column=0, sticky="nw")
        self.persona_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.persona_var, width=40).grid(row=2, column=1, pady=3)
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=8)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Add", command=self._add).pack(side="right", padx=6)

    def _add(self) -> None:
        arch_id = self.id_var.get().strip().lower().replace(" ", "-")
        name = self.name_var.get().strip() or arch_id
        persona = self.persona_var.get().strip()
        if not arch_id:
            messagebox.showerror("Missing id", "Give the archetype an id.", parent=self)
            return
        self.on_add(arch_id, name, persona)
        self.destroy()


class OptionsDialog(tk.Toplevel):
    def __init__(self, parent, cfg: ProjectConfig, toml_path: Path, on_saved=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.toml_path = toml_path
        self.on_saved = on_saved
        self.title(f"Game options — {cfg.name}")
        self.transient(parent)
        self.grab_set()
        self.geometry("680x460")
        self.check_vars: dict[str, tk.BooleanVar] = {}
        self._build()

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Game tone (applied to all agents):").grid(row=0, column=0, sticky="w")
        self.tone_var = tk.StringVar(value=self.cfg.tone)
        ttk.Entry(frm, textvariable=self.tone_var).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        frm.columnconfigure(1, weight=1)

        ttk.Label(
            frm, text="Archetypes (check = add as an agent):"
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(14, 2))

        listwrap = ttk.Frame(frm)
        listwrap.grid(row=2, column=0, columnspan=2, sticky="nsew")
        frm.rowconfigure(2, weight=1)
        canvas = tk.Canvas(listwrap, height=260, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(listwrap, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._populate_list()

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="we", pady=10)
        ttk.Button(btns, text="New archetype…", command=self._new_archetype).pack(side="left")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Save", command=self._save).pack(side="right", padx=6)

    def _populate_list(self) -> None:
        for w in self.inner.winfo_children():
            w.destroy()
        self.check_vars = {}
        row = 0
        for arch_id, arch in sorted(self.cfg.all_archetypes().items()):
            var = tk.BooleanVar(value=self.cfg.agent_for_archetype(arch_id) is not None)
            self.check_vars[arch_id] = var
            ttk.Checkbutton(
                self.inner,
                text=f"{arch.name} — {arch.persona}",
                variable=var,
                wraplength=520,
                justify="left",
            ).grid(row=row, column=0, sticky="w", pady=1)
            if arch_id in self.cfg.archetypes:  # custom → deletable
                ttk.Button(
                    self.inner,
                    text="X",
                    width=3,
                    command=lambda a=arch_id: self._delete_archetype(a),
                ).grid(row=row, column=1, padx=4)
            row += 1

    def _new_archetype(self) -> None:
        NewArchetypeDialog(self, self._add_archetype)

    def _add_archetype(self, arch_id: str, name: str, persona: str) -> None:
        self.cfg.archetypes[arch_id] = Archetype(arch_id, name or arch_id, persona)
        self.cfg.enable_archetype(arch_id)
        self._populate_list()

    def _delete_archetype(self, arch_id: str) -> None:
        self.cfg.archetypes.pop(arch_id, None)
        self.cfg.disable_archetype(arch_id)
        self._populate_list()

    def _save(self) -> None:
        self.cfg.tone = self.tone_var.get().strip()
        for arch_id, var in self.check_vars.items():
            self.cfg.set_archetype_enabled(arch_id, var.get())
        self.toml_path.write_text(dump_config(self.cfg), encoding="utf-8")
        self.destroy()
        if self.on_saved:
            self.on_saved()


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
        self.toml_path: Path | None = None
        self._build()
        root.after(100, self._poll)

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        menubar = tk.Menu(self.root)
        opts = tk.Menu(menubar, tearoff=0)
        opts.add_command(label="Archetypes & tone…", command=self.open_options)
        menubar.add_cascade(label="Options", menu=opts)
        self.root.config(menu=menubar)

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
        self.toml_path = toml
        self.fw = load_framework(toml.parent / self.cfg.framework_path)
        return toml.parent

    # ---- options menu -----------------------------------------------------
    def open_options(self) -> None:
        if not self.cfg:
            try:
                self._load()
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Load error", str(e))
                return
        OptionsDialog(self.root, self.cfg, self.toml_path, on_saved=self._on_options_saved)

    def _on_options_saved(self) -> None:
        self.status.set(f"Options saved → {self.toml_path}")
        self._append(f"[options] saved → {self.toml_path}")

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
