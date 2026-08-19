"""Command-line interface for Haija."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import PROVIDER_DEFAULTS, ProjectConfig
from .engine import Engine, render_export, run_game
from .framework import load_framework, validate_framework
from .generate import generate_framework
from .logging_setup import setup_logging
from .project import new_project, projects_root
from .provider import ProviderError, create_provider


def _resolve_project(p: str) -> Path:
    path = Path(p)
    if path.is_dir():
        path = path / "haija.toml"
    if not path.exists():
        print(f"error: no haija.toml found at {path}", file=sys.stderr)
        raise SystemExit(1)
    return path


def cmd_new(args: argparse.Namespace) -> int:
    dest = Path(args.dir) if args.dir else projects_root()
    try:
        root = new_project(args.name, dest)
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Created Haija project '{args.name}' at {root}")
    print("Next: set OPENROUTER_API_KEY, then `haija generate \"<prompt>\"` and `haija run`.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    provider = create_provider(cfg.model)

    def _observer(ev):
        if ev["type"] == "generate_start":
            print(f"Generating framework for '{cfg.name}'…", file=sys.stderr)
            print("  ", end="", flush=True)
        elif ev["type"] == "generate_stream":
            if args.no_stream:
                return
            # Print a dot per ~80 chars to show progress
            print(".", end="", flush=True)
        elif ev["type"] == "generate_validation_failed":
            print(" validation failed.", file=sys.stderr)
            for error in ev.get("errors", []):
                print(f"    ✗ {error}", file=sys.stderr)
        elif ev["type"] == "generate_repair_start":
            print(
                f"  Attempting automatic repair "
                f"({ev['attempt']}/{ev['max_attempts']})…",
                file=sys.stderr,
            )
            print("  ", end="", flush=True)
        elif ev["type"] == "generate_done":
            print(" done.", file=sys.stderr)

    try:
        fw = generate_framework(provider, args.prompt, name=cfg.name, observer=_observer)
    except (ProviderError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = toml.parent / cfg.framework_path
    out.write_text(json.dumps(fw.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote framework '{fw.name}' → {out}")
    print(f"  {len(fw.actions)} action(s), {len(fw.rules)} rule(s), max_turns={fw.turn.max_turns}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    if not cfg.agents:
        print("error: no agents configured in haija.toml", file=sys.stderr)
        return 1
    fw = load_framework(toml.parent / cfg.framework_path)
    errors, warnings = validate_framework(fw)
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if args.turns:
        fw.turn.max_turns = args.turns
    provider = create_provider(cfg.model)
    engine = Engine(fw, [a.name for a in cfg.agents], max_steps_per_turn=cfg.max_steps_per_turn)
    try:
        msg = run_game(engine, provider, cfg)
    except ProviderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"\n{msg}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    fw = load_framework(toml.parent / cfg.framework_path)
    errors, warnings = validate_framework(fw)
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: framework '{fw.name}' — {len(fw.actions)} action(s), "
        f"{len(fw.rules)} rule(s), {len(cfg.agents)} agent(s)"
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    base = toml.parent
    fw = load_framework(base / cfg.framework_path)
    run_path = base / "run.json"
    if not run_path.exists():
        print(f"error: no run.json found at {run_path}. Run a game first.", file=sys.stderr)
        return 1
    run_log = json.loads(run_path.read_text(encoding="utf-8"))
    state_path = base / cfg.state_path
    final_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    )
    out = Path(args.output) if args.output else base / "haija-export.txt"
    text = render_export(
        fw.name,
        fw.description,
        fw.objective,
        fw.rules,
        fw.win_conditions,
        fw.lose_conditions,
        [a.name for a in cfg.agents],
        run_log,
        final_state,
    )
    out.write_text(text, encoding="utf-8")
    print(f"Exported → {out}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import main as gui_main
    except Exception as e:  # noqa: BLE001
        print(f"error: could not start the GUI: {e}", file=sys.stderr)
        return 1
    return gui_main()


def cmd_archetypes(args: argparse.Namespace) -> int:
    from .archetypes import BUILTIN_ARCHETYPES

    built = list(BUILTIN_ARCHETYPES.values())
    p = Path(args.project)
    toml = p / "haija.toml" if p.is_dir() else p
    cfg = None
    if toml.exists():
        try:
            cfg = ProjectConfig.load(toml)
        except Exception:
            pass
    print("Built-in archetypes:")
    for arch in built:
        mark = "✓" if (cfg and cfg.agent_for_archetype(arch.id)) else " "
        print(f"  [{mark}] {arch.name:16s} ({arch.id}) — {arch.persona}")
    if cfg and cfg.archetypes:
        print("\nCustom archetypes (from haija.toml):")
        for arch in cfg.archetypes.values():
            mark = "✓" if cfg.agent_for_archetype(arch.id) else " "
            print(f"  [{mark}] {arch.name:16s} ({arch.id}) — {arch.persona}")
    if cfg and cfg.agents:
        print(f"\nActive agents ({len(cfg.agents)}):")
        for a in cfg.agents:
            arch = f"({a.archetype})" if a.archetype else ""
            print(f"  • {a.name} {arch}")
            if a.description:
                print(f"    {a.description}")
    if cfg and cfg.tone:
        print(f"\nGame tone: {cfg.tone}")
    return 0


def _split_ids(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _print_options(cfg: ProjectConfig) -> None:
    from .archetypes import BUILTIN_ARCHETYPES

    print(f"Project:   {cfg.name}")
    print(f"Tone:      {cfg.tone or '(none)'}")
    print(f"Model:     {cfg.model.provider} / {cfg.model.model}")
    print(
        f"Runtime:   framework={cfg.framework_path} state={cfg.state_path} "
        f"max_steps_per_turn={cfg.max_steps_per_turn}"
    )
    print(f"Logging:   level={cfg.general.log_level} file={cfg.general.log_file or '~/.haija/haija.log'}")
    print(f"\nAgents ({len(cfg.agents)}):")
    for a in cfg.agents:
        arch = f" ({a.archetype})" if a.archetype else ""
        desc = f" — {a.description}" if a.description else ""
        model = f" [model={a.model}]" if a.model else ""
        think = f" [thinking={a.thinking}]" if a.thinking else ""
        print(f"  • {a.name}{arch}{desc}{model}{think}")
    built = list(BUILTIN_ARCHETYPES.values())
    custom = list(cfg.archetypes.values())
    print(f"\nArchetypes ({len(built)} built-in, {len(custom)} custom):")
    for arch in built:
        mark = "✓" if cfg.agent_for_archetype(arch.id) else " "
        print(f"  [{mark}] {arch.name:16s} ({arch.id}) — {arch.persona}")
    for arch in custom:
        mark = "✓" if cfg.agent_for_archetype(arch.id) else " "
        print(f"  [{mark}] {arch.name:16s} ({arch.id}) — {arch.persona}  [custom]")


def cmd_options(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    changed = False

    if args.tone is not None:
        cfg.set_tone(args.tone)
        changed = True
    if args.enable:
        for i in _split_ids(args.enable):
            cfg.set_archetype_enabled(i, True)
        changed = True
    if args.disable:
        for i in _split_ids(args.disable):
            cfg.set_archetype_enabled(i, False)
        changed = True
    if args.add_archetype:
        cfg.add_archetype(*args.add_archetype)
        changed = True
    if args.remove_archetype:
        cfg.remove_archetype(args.remove_archetype)
        changed = True
    if args.provider:
        provider_name = args.provider.strip().lower()
        if provider_name not in PROVIDER_DEFAULTS:
            print(
                "error: provider must be one of " + ", ".join(PROVIDER_DEFAULTS),
                file=sys.stderr,
            )
            return 1
        cfg.model.provider = provider_name
        cfg.model.model, cfg.model.base_url, cfg.model.api_key_env = (
            PROVIDER_DEFAULTS[provider_name]
        )
        changed = True
    if args.model:
        cfg.model.model = args.model
        changed = True
    if args.base_url is not None:
        cfg.model.base_url = args.base_url
        changed = True
    if args.api_key_env:
        cfg.model.api_key_env = args.api_key_env
        changed = True
    if args.agent_thinking:
        for item in _split_ids(args.agent_thinking):
            name, _, level = item.partition("=")
            name = name.strip()
            if not name:
                continue
            level = level.strip().lower()
            cfg.set_agent_thinking(name, None if level in ("", "default") else level)
        changed = True
    if args.log_level:
        cfg.general.log_level = args.log_level.strip().lower()
        changed = True
    if args.log_file:
        cfg.general.log_file = args.log_file.strip()
        changed = True

    if changed:
        cfg.save()
        print(f"Saved options → {toml}\n")
    _print_options(cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Setup logging early — defaults to info, overridden by project config later.
    setup_logging()

    p = argparse.ArgumentParser(
        prog="haija",
        description="Haija — an AI game engine where the agents are the game.",
    )
    p.add_argument("--version", action="version", version=f"haija {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    n = sub.add_parser("new", help="create a new project")
    n.add_argument("name")
    n.add_argument("--dir", default=None, help="parent directory (default: <haija home>/projects)")
    n.set_defaults(func=cmd_new)

    g = sub.add_parser("generate", help="generate a framework from a prompt")
    g.add_argument("prompt")
    g.add_argument("--project", default=".", help="project directory or haija.toml path")
    g.add_argument("--no-stream", action="store_true", help="don't show progress dots")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run the game")
    r.add_argument("--project", default=".", help="project directory or haija.toml path")
    r.add_argument("--turns", type=int, default=None, help="override max_turns")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="validate a project's framework")
    v.add_argument("--project", default=".", help="project directory or haija.toml path")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("export", help="export the last run as a human-readable .txt")
    e.add_argument("--project", default=".", help="project directory or haija.toml path")
    e.add_argument("-o", "--output", default=None, help="output .txt path (default: haija-export.txt)")
    e.set_defaults(func=cmd_export)

    gui = sub.add_parser("gui", help="launch the graphical interface")
    gui.set_defaults(func=cmd_gui)

    arc = sub.add_parser("archetypes", help="list built-in and project archetypes")
    arc.add_argument("--project", default=".", help="project directory or haija.toml path (optional)")
    arc.set_defaults(func=cmd_archetypes)

    o = sub.add_parser("options", help="view and edit game options (tone, archetypes, model)")
    o.add_argument("--project", default=".", help="project directory or haija.toml path")
    o.add_argument("--tone", default=None, help="set the game tone (use empty string to clear)")
    o.add_argument("--enable", default=None, metavar="IDS", help="comma-separated archetype ids to enable")
    o.add_argument("--disable", default=None, metavar="IDS", help="comma-separated archetype ids to disable")
    o.add_argument("--add-archetype", nargs=3, metavar=("ID", "NAME", "PERSONA"), default=None, help="add a custom archetype")
    o.add_argument("--remove-archetype", default=None, metavar="ID", help="remove a custom archetype")
    o.add_argument("--model", default=None, help="set the default model")
    o.add_argument("--base-url", default=None, help="set the API base URL")
    o.add_argument(
        "--provider",
        default=None,
        choices=list(PROVIDER_DEFAULTS),
        help="set the model provider",
    )
    o.add_argument("--api-key-env", default=None, help="set the API key env var")
    o.add_argument("--agent-thinking", default=None, metavar="NAME=LEVEL[,...]", help="set per-agent thinking (off|low|medium|high|default), comma-separated")
    o.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error"], help="set log level")
    o.add_argument("--log-file", default=None, metavar="PATH", help="set log file path")
    o.set_defaults(func=cmd_options)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
