"""Command-line interface for Haija."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ProjectConfig
from .engine import Engine, run_game
from .framework import load_framework
from .generate import generate_framework
from .project import new_project
from .provider import ChatProvider, ProviderError


def _resolve_project(p: str) -> Path:
    path = Path(p)
    if path.is_dir():
        path = path / "haija.toml"
    if not path.exists():
        print(f"error: no haija.toml found at {path}", file=sys.stderr)
        raise SystemExit(1)
    return path


def cmd_new(args: argparse.Namespace) -> int:
    root = new_project(args.name, Path(args.dir))
    print(f"Created Haija project '{args.name}' at {root}")
    print("Next: set OPENROUTER_API_KEY, then `haija generate \"<prompt>\"` and `haija run`.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    provider = ChatProvider(cfg.model)
    try:
        fw = generate_framework(provider, args.prompt)
    except ProviderError as e:
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
    if args.turns:
        fw.turn.max_turns = args.turns
    provider = ChatProvider(cfg.model)
    engine = Engine(fw, [a.name for a in cfg.agents], max_steps_per_turn=cfg.max_steps_per_turn)
    try:
        run_game(engine, provider, cfg)
    except ProviderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    toml = _resolve_project(args.project)
    cfg = ProjectConfig.load(toml)
    fw = load_framework(toml.parent / cfg.framework_path)
    print(
        f"OK: framework '{fw.name}' — {len(fw.actions)} action(s), "
        f"{len(fw.rules)} rule(s), {len(cfg.agents)} agent(s)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="haija",
        description="Haija — an AI game engine where the agents are the game.",
    )
    p.add_argument("--version", action="version", version=f"haija {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    n = sub.add_parser("new", help="create a new project")
    n.add_argument("name")
    n.add_argument("--dir", default=".", help="parent directory (default: .)")
    n.set_defaults(func=cmd_new)

    g = sub.add_parser("generate", help="generate a framework from a prompt")
    g.add_argument("prompt")
    g.add_argument("--project", default=".", help="project directory or haija.toml path")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="run the game")
    r.add_argument("--project", default=".", help="project directory or haija.toml path")
    r.add_argument("--turns", type=int, default=None, help="override max_turns")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="validate a project's framework")
    v.add_argument("--project", default=".", help="project directory or haija.toml path")
    v.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
