"""Tiny templating + path resolution for the effect DSL."""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def _lookup(root: Any, dotted: str) -> Any:
    cur = root
    for seg in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur[seg]
        elif isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            raise KeyError(dotted)
    return cur


def render(template: str, ctx: dict[str, Any]) -> str:
    """Render ``{{ expr }}`` tokens. Returns input unchanged if not a string."""
    if not isinstance(template, str):
        return template

    def repl(m: re.Match[str]) -> str:
        expr = m.group(1).strip()
        if "." not in expr:
            val = ctx.get(expr)
        else:
            root_name, _, rest = expr.partition(".")
            root = ctx.get(root_name)
            try:
                val = _lookup(root, rest) if root is not None else None
            except (KeyError, IndexError, ValueError, TypeError):
                val = None
        return str(val) if val is not None else ""

    return _TOKEN_RE.sub(repl, template)


def split_path(path: str) -> list[str]:
    """Split a dot-path on dots that are *outside* ``{{ }}`` templates."""
    segs: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in path:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "." and depth == 0:
            segs.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    segs.append("".join(buf))
    return segs


def resolve_path(path: str, ctx: dict[str, Any]) -> list[str]:
    """Split a dot-path and render any template segments."""
    return [render(seg, ctx) if "{{" in seg else seg for seg in split_path(path)]


def resolve_value(value: Any, ctx: dict[str, Any]) -> Any:
    """Render a value if it's a template string, otherwise return it as-is."""
    if isinstance(value, str) and "{{" in value:
        return render(value, ctx)
    return value
