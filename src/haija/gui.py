"""Haija's graphical interface — full feature parity with the CLI.

Every CLI action is available here (new / open / generate / validate / run /
export / options), plus a cleaner toolbar + sidebar layout. Stdlib-only
(``tkinter``). Launched via ``haija gui``.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import Tk, filedialog, messagebox, ttk

from .archetypes import Archetype
from .config import ProjectConfig
from .engine import Engine, run_game
from .framework import load_framework
from .generate import generate_framework
from .project import new_project
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


class NewProjectDialog(tk.Toplevel):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.result: tuple[str, str] | None = None
        self.title("New project")
        self.transient(parent)
        self.grab_set()
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Name:").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var, width=30).grid(row=0, column=1, pady=3, sticky="we")
        ttk.Label(frm, text="Directory:").grid(row=1, column=0, sticky="w")
        self.dir_var = tk.StringVar(value=".")
        ttk.Entry(frm, textvariable=self.dir_var, width=30).grid(row=1, column=1, pady=3, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._browse).grid(row=1, column=2, padx=4)
        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=3, sticky="e", pady=8)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Create", command=self._ok).pack(side="right", padx=6)

    def _browse(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.dir_var.set(d)

    def _ok(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Missing name", "Give the project a name.", parent=self)
            return
        self.result = (name, self.dir_var.get().strip() or ".")
        self.destroy()


class PromptDialog(tk.Toplevel):
    def __init__(self, parent, title: str, label: str) -> None:
        super().__init__(parent)
        self.result: str | None = None
        self.title(title)
        self.transient(parent)
        self.grab_set()
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=label).pack(anchor="w")
        self.text = tk.Text(frm, height=6, width=60, wrap="word")
        self.text.pack(fill="both", expand=True, pady=6)
        btns = ttk.Frame(frm)
        btns.pack(anchor="e")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="OK", command=self._ok).pack(side="right", padx=6)

    def _ok(self) -> None:
        self.result = self.text.get("1.0", "end").strip()
        self.destroy()


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
        self.geometry("700x500")
        self.check_vars: dict[str, tk.BooleanVar] = {}
        self._build()

    def _build(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        agents_tab = ttk.Frame(nb, padding=10)
        model_tab = ttk.Frame(nb, padding=10)
        nb.add(agents_tab, text="Agents & tone")
        nb.add(model_tab, text="Model")
        self._build_agents_tab(agents_tab)
        self._build_model_tab(model_tab)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Save", command=self._save).pack(side="right", padx=6)

    def _build_agents_tab(self, frm: ttk.Frame) -> None:
        ttk.Label(frm, text="Game tone (applied to all agents):").grid(row=0, column=0, sticky="w")
        self.tone_var = tk.StringVar(value=self.cfg.tone)
        ttk.Entry(frm, textvariable=self.tone_var).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Archetypes (check = add as an agent):").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(14, 2)
        )
        listwrap = ttk.Frame(frm)
        listwrap.grid(row=2, column=0, columnspan=2, sticky="nsew")
        frm.rowconfigure(2, weight=1)
        canvas = tk.Canvas(listwrap, height=240, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(listwrap, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._populate_list()

        ttk.Button(frm, text="New archetype…", command=self._new_archetype).grid(
            row=3, column=0, sticky="w", pady=8
        )

    def _build_model_tab(self, frm: ttk.Frame) -> None:
        fields = [
            ("Provider", "provider_var", self.cfg.model.provider),
            ("Model", "model_var", self.cfg.model.model),
            ("Base URL", "base_url_var", self.cfg.model.base_url),
            ("API key env var", "api_key_env_var", self.cfg.model.api_key_env),
            ("API key (optional)", "api_key_var", self.cfg.model.api_key or ""),
        ]
        for i, (label, var_name, value) in enumerate(fields):
            ttk.Label(frm, text=label + ":").grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=value or "")
            setattr(self, var_name, var)
            ttk.Entry(frm, textvariable=var, width=46).grid(row=i, column=1, sticky="we", padx=6, pady=4)
        frm.columnconfigure(1, weight=1)
        ttk.Label(
            frm,
            text="The API key is read from the env var by default; hardcode it only if you must.",
            wraplength=420,
            foreground="#666",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=8)

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
                    self.inner, text="X", width=3, command=lambda a=arch_id: self._delete_archetype(a)
                ).grid(row=row, column=1, padx=4)
            row += 1

    def _new_archetype(self) -> None:
        NewArchetypeDialog(self, self._add_archetype)

    def _add_archetype(self, arch_id: str, name: str, persona: str) -> None:
        self.cfg.add_archetype(arch_id, name, persona)
        self.cfg.enable_archetype(arch_id)
        self._populate_list()

    def _delete_archetype(self, arch_id: str) -> None:
        self.cfg.remove_archetype(arch_id)
        self._populate_list()

    def _save(self) -> None:
        self.cfg.tone = self.tone_var.get().strip()
        for arch_id, var in self.check_vars.items():
            self.cfg.set_archetype_enabled(arch_id, var.get())
        self.cfg.model.provider = self.provider_var.get().strip() or "openrouter"
        self.cfg.model.model = self.model_var.get().strip() or self.cfg.model.model
        self.cfg.model.base_url = self.base_url_var.get().strip() or self.cfg.model.base_url
        self.cfg.model.api_key_env = self.api_key_env_var.get().strip() or self.cfg.model.api_key_env
        self.cfg.model.api_key = self.api_key_var.get().strip() or None
        self.cfg.save(self.toml_path)
        self.destroy()
        if self.on_saved:
            self.on_saved()


class HaijaApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        root.title("Haija — AI game engine")
        root.geometry("1000x680")
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.cfg: ProjectConfig | None = None
        self.fw = None
        self.engine: Engine | None = None
        self.worker: threading.Thread | None = None
        self.toml_path: Path | None = None
        self.project_dir: Path = Path(".")
        self.busy = False
        self._pending_export = False
        self._build()
        root.after(100, self._poll)

    # ---- layout -----------------------------------------------------------
    def _build(self) -> None:
        self._build_menu()
        self._build_toolbar()

        status = ttk.Frame(self.root)
        status.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Ready — open a project to begin")
        ttk.Label(status, textvariable=self.status, padding=4, relief="sunken").pack(fill="x")

        content = ttk.Frame(self.root, padding=8)
        content.pack(side="top", fill="both", expand=True)
        self._build_sidebar(content)
        self._build_log(content)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Project…", command=self._new_project)
        file_menu.add_command(label="Open Project…", command=self._open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Export Run…", command=self._export)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(label="Generate Framework…", command=self._generate)
        game_menu.add_command(label="Validate", command=self._validate)
        game_menu.add_separator()
        game_menu.add_command(label="Run", command=self._run)
        menubar.add_cascade(label="Game", menu=game_menu)

        opts_menu = tk.Menu(menubar, tearoff=0)
        opts_menu.add_command(label="Archetypes & tone…", command=self.open_options)
        menubar.add_cascade(label="Options", menu=opts_menu)
        self.root.config(menu=menubar)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side="top", fill="x")
        self.new_btn = ttk.Button(bar, text="New", command=self._new_project)
        self.open_btn = ttk.Button(bar, text="Open", command=self._open_project)
        self.gen_btn = ttk.Button(bar, text="Generate", command=self._generate)
        self.val_btn = ttk.Button(bar, text="Validate", command=self._validate)
        self.run_btn = ttk.Button(bar, text="Run", command=self._run)
        self.export_btn = ttk.Button(bar, text="Export", command=self._export, state="disabled")
        for i, b in enumerate(
            (self.new_btn, self.open_btn, self.gen_btn, self.val_btn, self.run_btn, self.export_btn)
        ):
            b.pack(side="left", padx=2)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        side = ttk.Frame(parent, padding=(0, 0, 8, 0))
        side.pack(side="left", fill="y")
        ttk.Label(side, text="PROJECT", font=("", 9, "bold")).pack(anchor="w")
        self.proj_name = ttk.Label(side, text="(none)", wraplength=200)
        self.proj_name.pack(anchor="w")
        ttk.Label(side, text="Tone:").pack(anchor="w", pady=(10, 0))
        self.tone_label = ttk.Label(side, text="(none)", wraplength=200)
        self.tone_label.pack(anchor="w")
        ttk.Label(side, text="Model:").pack(anchor="w", pady=(10, 0))
        self.model_label = ttk.Label(side, text="(none)", wraplength=200)
        self.model_label.pack(anchor="w")
        ttk.Label(side, text="Agents:").pack(anchor="w", pady=(10, 0))
        listwrap = ttk.Frame(side)
        listwrap.pack(anchor="w", fill="y", expand=True, pady=2)
        self.agents_list = tk.Listbox(listwrap, height=12, width=26)
        sb = ttk.Scrollbar(listwrap, orient="vertical", command=self.agents_list.yview)
        self.agents_list.configure(yscrollcommand=sb.set)
        self.agents_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_log(self, parent: ttk.Frame) -> None:
        main = ttk.Frame(parent)
        main.pack(side="left", fill="both", expand=True)
        self.log = tk.Text(main, wrap="word", state="disabled", font=("Consolas", 10))
        sb = ttk.Scrollbar(main, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ---- helpers ----------------------------------------------------------
    def _append(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.new_btn, self.open_btn, self.gen_btn, self.val_btn, self.run_btn):
            b.configure(state=state)

    def _refresh_sidebar(self) -> None:
        if not self.cfg:
            self.proj_name.configure(text="(none)")
            self.tone_label.configure(text="(none)")
            self.model_label.configure(text="(none)")
            self.agents_list.delete(0, "end")
            return
        self.proj_name.configure(text=f"{self.cfg.name}\n{self.project_dir}")
        self.tone_label.configure(text=self.cfg.tone or "(none)")
        self.model_label.configure(text=f"{self.cfg.model.provider}\n{self.cfg.model.model}")
        self.agents_list.delete(0, "end")
        for a in self.cfg.agents:
            arch = a.archetype or "custom"
            self.agents_list.insert("end", f"{a.name} ({arch})")

    def _load(self) -> Path:
        toml = self.project_dir / "haija.toml"
        if not toml.exists():
            raise FileNotFoundError(f"no haija.toml found at {self.project_dir}")
        self.cfg = ProjectConfig.load(toml)
        self.toml_path = toml
        self.fw = load_framework(toml.parent / self.cfg.framework_path)
        self._refresh_sidebar()
        return toml.parent

    def _ensure_loaded(self) -> bool:
        if self.cfg:
            return True
        try:
            self._load()
            return True
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load error", str(e))
            return False

    # ---- actions ----------------------------------------------------------
    def _new_project(self) -> None:
        d = NewProjectDialog(self.root)
        self.root.wait_window(d)
        if not d.result:
            return
        name, directory = d.result
        try:
            root = new_project(name, Path(directory))
            self.project_dir = root
            self._load()
            self._clear_log()
            self._append(f"[new] created project '{name}' at {root}")
            self.status.set(f"Created {name}")
        except FileExistsError:
            messagebox.showerror("New project", f"'{name}' already exists.")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("New project", str(e))

    def _open_project(self) -> None:
        d = filedialog.askdirectory()
        if not d:
            return
        self.project_dir = Path(d)
        try:
            self._load()
            self._clear_log()
            self._append(f"[open] loaded project '{self.cfg.name}'")
            self.status.set(f"Loaded {self.cfg.name}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Open project", str(e))

    def _validate(self) -> None:
        if not self._ensure_loaded():
            return
        try:
            self.fw = load_framework(self.toml_path.parent / self.cfg.framework_path)
            self._append(
                f"[validate] OK — framework '{self.fw.name}' "
                f"({len(self.fw.actions)} actions, {len(self.cfg.agents)} agents)"
            )
            messagebox.showinfo(
                "Validate",
                f"OK — framework '{self.fw.name}'\n"
                f"{len(self.fw.actions)} action(s), {len(self.cfg.agents)} agent(s)",
            )
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Validate", str(e))

    def _generate(self) -> None:
        if not self._ensure_loaded():
            return
        d = PromptDialog(self.root, "Generate framework", "Describe your game:")
        self.root.wait_window(d)
        if not d.result:
            return
        prompt = d.result
        self._set_busy(True)
        self._pending_export = False
        self.status.set("Generating framework…")
        self._append(f"[generate] prompt: {prompt}")

        def work() -> None:
            try:
                provider = ChatProvider(self.cfg.model)
                fw = generate_framework(provider, prompt)
                out = self.toml_path.parent / self.cfg.framework_path
                out.write_text(json.dumps(fw.to_dict(), indent=2) + "\n", encoding="utf-8")
                self.fw = fw
                self.q.put({"type": "info", "message": f"Wrote framework '{fw.name}' → {out}"})
            except Exception as e:  # noqa: BLE001
                self.q.put({"type": "error", "message": str(e)})
            finally:
                self.q.put({"type": "done"})

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        if self.busy:
            return
        if not self._ensure_loaded():
            return
        self._clear_log()
        self._set_busy(True)
        self._pending_export = True
        self.export_btn.configure(state="disabled")
        self.status.set("Running…")

        def observer(event: dict) -> None:
            self.q.put(event)

        self.worker = threading.Thread(
            target=self._run_thread, args=(self.toml_path.parent, observer), daemon=True
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

    # ---- options ----------------------------------------------------------
    def open_options(self) -> None:
        if not self._ensure_loaded():
            return
        OptionsDialog(self.root, self.cfg, self.toml_path, on_saved=self._on_options_saved)

    def _on_options_saved(self) -> None:
        self._refresh_sidebar()
        self.status.set(f"Options saved → {self.toml_path}")
        self._append(f"[options] saved → {self.toml_path}")

    # ---- export -----------------------------------------------------------
    def _export(self) -> None:
        if not self.engine:
            messagebox.showinfo("Nothing to export", "Run a game first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="haija-export.txt")
        if not path:
            return
        Path(path).write_text(self.engine.export_text(), encoding="utf-8")
        self.status.set(f"Exported → {path}")

    # ---- event loop -------------------------------------------------------
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
            self._set_busy(False)
        elif t == "done":
            self._set_busy(False)
            if self._pending_export:
                self.export_btn.configure(state="normal")
                self.status.set("Finished — click Export to save the run")
            else:
                self.status.set("Done")
        elif t == "info":
            self.status.set(ev["message"])
        else:
            self._append(format_event(ev))


def main() -> int:
    root = Tk()
    HaijaApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
