"""Project scaffolding: ``haija new``."""

from __future__ import annotations

import json
from pathlib import Path

from .config import default_config
from .framework import SCHEMA_VERSION


def new_project(name: str, dest: Path) -> Path:
    root = dest / name
    root.mkdir(parents=True, exist_ok=False)
    (root / "haija.toml").write_text(default_config(name), encoding="utf-8")
    (root / "framework.json").write_text(_empty_framework(name), encoding="utf-8")
    (root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\nstate.json\ntranscript.json\nrun.json\nhaija-export.txt\n.venv/\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(_project_readme(name), encoding="utf-8")
    return root


def _empty_framework(name: str) -> str:
    return (
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "name": name,
                "description": "Describe your game world here (or run `haija generate`).",
                "objective": "",
                "rules": [],
                "win_conditions": [],
                "lose_conditions": [],
                "initial_state": {},
                "actions": [],
                "turn": {"order": "round_robin", "max_turns": 20},
            },
            indent=2,
        )
        + "\n"
    )


def _project_readme(name: str) -> str:
    return f"""# {name}

Generated with [Haija](https://github.com/UniqueName54321/haija).

## Quick start

1. Set your API key: `export OPENROUTER_API_KEY=sk-or-...`
2. (Optional) Generate a framework from a prompt:
   `haija generate "two rival AIs negotiate a trade deal"`
3. Edit `haija.toml` to customize your agents (names, descriptions, models).
4. Run the game: `haija run`

See the Haija README for the full framework-file spec.
"""
