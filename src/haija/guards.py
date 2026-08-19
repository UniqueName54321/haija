"""Declarative guards: action preconditions and win/lose/draw conditions.

A *guard* is a small JSON predicate evaluated against game state. The engine
uses guards two ways:

- **Rule guards** — an action's ``guard`` must pass before its effects apply,
  so illegal moves are rejected by the engine (the source of truth), not by the
  model.
- **The deterministic judge** — a framework's ``judge.win`` / ``lose`` / ``draw``
  guards are checked after every action; when one passes, the engine declares
  the outcome itself instead of trusting an agent to do it.

Ops: ``eq``, ``neq``, ``gt``, ``gte``, ``lt``, ``lte``, ``in``, ``not_in``,
``exists``, ``not_exists``. A guard may be negated with ``{"not": {...}}`` and a
list of guards is an implicit AND.
"""

from __future__ import annotations

from typing import Any

from .templating import resolve_path, resolve_value


class GuardError(ValueError):
    """Raised when a guard uses an unknown op or is malformed."""


def _navigate(state: Any, segs: list[str]) -> tuple[bool, Any]:
    """Navigate ``segs`` into ``state``. Returns ``(found, value)``."""
    cur = state
    for seg in segs:
        if isinstance(cur, dict):
            if seg not in cur:
                return False, None
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except (ValueError, TypeError):
                # Generated frameworks often express list membership as a
                # value-addressed path, e.g. hands.Alice.Y0. Treat a
                # non-numeric segment as the requested list item.
                try:
                    idx = cur.index(seg)
                except ValueError:
                    return False, None
            if idx < 0 or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


def _to_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _smart_eq(a: Any, b: Any) -> bool:
    """Equality that also treats ``3`` and ``"3"`` as equal (template-friendly)."""
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    return na is not None and nb is not None and na == nb


def evaluate_guard(guard: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate one guard. Returns ``(passed, human-readable reason)``."""
    if not isinstance(guard, dict):
        return False, "guard must be an object"
    if "not" in guard:
        passed, reason = evaluate_guard(guard["not"], ctx)
        return not passed, reason

    op = guard.get("op", "eq")
    if op not in ("eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "exists", "not_exists"):
        raise GuardError(f"unknown guard op: {op}")
    path = guard.get("path", "")
    segs = resolve_path(path, ctx)
    display = ".".join(segs)
    found, actual = _navigate(ctx["state"], segs)
    expected = resolve_value(guard.get("value"), ctx)

    if op in ("exists", "not_exists"):
        if op == "exists":
            return (found, f"{display} exists" if found else f"{display} is missing")
        return (not found, f"{display} is missing" if not found else f"{display} exists")

    if not found:
        return (False, f"{display} is missing")

    if op == "eq":
        return (_smart_eq(actual, expected), f"{display} == {expected!r} (got {actual!r})")
    if op == "neq":
        return (not _smart_eq(actual, expected), f"{display} != {expected!r} (got {actual!r})")
    if op in ("gt", "gte", "lt", "lte"):
        na, nb = _to_number(actual), _to_number(expected)
        if na is None or nb is None:
            return (False, f"cannot compare {display} ({actual!r}) with {expected!r}")
        if op == "gt":
            passed = na > nb
        elif op == "gte":
            passed = na >= nb
        elif op == "lt":
            passed = na < nb
        else:
            passed = na <= nb
        return (passed, f"{display} {op} {expected!r} (got {actual!r})")
    if op in ("in", "not_in"):
        try:
            contained = actual in expected
        except TypeError:
            return (False, f"cannot test membership for {display}")
        passed = contained if op == "in" else not contained
        return (passed, f"{display} {op} {expected!r} (got {actual!r})")
    raise GuardError(f"unknown guard op: {op}")


def evaluate_guards(guards: Any, ctx: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate a guard or list of guards (implicit AND). ``(passed, reason)``."""
    if not guards:
        return True, ""
    if isinstance(guards, dict):
        guards = [guards]
    for g in guards:
        passed, reason = evaluate_guard(g, ctx)
        if not passed:
            return False, reason
    return True, "all guards passed"
