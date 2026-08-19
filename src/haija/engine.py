"""The engine: owns game state (the "truth") and runs the turn loop."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import run_agent
from .config import ProjectConfig
from .framework import Action, Framework
from .templating import resolve_path, resolve_value


def _to_number(v: Any) -> int | float:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"cannot use '{v}' in an 'incr' effect") from None


@dataclass
class Engine:
    framework: Framework
    agents: list[str]
    max_steps_per_turn: int = 16
    state: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    outcome: str | None = None
    winner: str | None = None

    def __post_init__(self) -> None:
        if not self.state:
            self.state = copy.deepcopy(self.framework.initial_state)

    # ---- state access -----------------------------------------------------
    def public_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def get_state(self, paths: list[str] | None = None) -> dict[str, Any]:
        full = self.public_state()
        if not paths:
            return full
        out: dict[str, Any] = {}
        for p in paths:
            cur: Any = full
            for seg in p.split("."):
                if isinstance(cur, dict) and seg in cur:
                    cur = cur[seg]
                elif isinstance(cur, list) and seg.lstrip("-").isdigit():
                    cur = cur[int(seg)]
                else:
                    cur = None
                    break
            out[p] = cur
        return out

    # ---- effect application ----------------------------------------------
    def _ctx(self, actor: str, params: dict[str, Any]) -> dict[str, Any]:
        mark = actor
        marks = self.state.get("marks")
        if isinstance(marks, dict) and actor in marks:
            mark = marks[actor]
        return {
            "actor": actor,
            "mark": mark,
            "params": params,
            "state": self.state,
            "turn": self.turn,
            "now": int(time.time()),
        }

    def apply_action(
        self, action: Action, actor: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        ctx = self._ctx(actor, params)
        results = []
        for eff in action.effects:
            results.append(self._apply_one(eff, ctx))
        self.history.append(
            {
                "turn": self.turn,
                "actor": actor,
                "action": action.name,
                "params": params,
                "effects": results,
            }
        )
        return {"ok": True, "effects": results, "state": self.public_state()}

    def _apply_one(self, eff: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        op = eff.get("op", "set")
        path = eff.get("path", "")
        raw = eff.get("value")
        segs = resolve_path(path, ctx)
        value = resolve_value(raw, ctx)

        if op == "set":
            parent, key = self._parent_key(segs)
            parent[key] = value
            return {"op": "set", "path": path, "value": value}
        if op == "incr":
            parent, key = self._parent_key(segs)
            delta = _to_number(value)
            parent[key] = parent[key] + delta
            return {"op": "incr", "path": path, "delta": delta}
        if op == "append":
            target = self._navigate(segs)
            target.append(value)
            return {"op": "append", "path": path, "value": value}
        if op == "remove":
            parent, key = self._parent_key(segs)
            if isinstance(parent, list):
                del parent[key]
            else:
                parent.pop(key, None)
            return {"op": "remove", "path": path}
        raise ValueError(f"unknown effect op: {op}")

    def _navigate(self, segs: list[str]) -> Any:
        cur: Any = self.state
        for seg in segs:
            cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
        return cur

    def _parent_key(self, segs: list[str]) -> tuple[Any, Any]:
        parent = self._navigate(segs[:-1])
        key: Any = segs[-1]
        if isinstance(parent, list):
            key = int(key)
        return parent, key

    # ---- tool dispatch ----------------------------------------------------
    def dispatch_tool(self, name: str, args: dict[str, Any], actor: str) -> str:
        if name == "get_state":
            return json.dumps(self.get_state(args.get("paths")), indent=2)
        if name == "end_turn":
            self.messages.append(
                {"from": actor, "type": "turn_end", "text": args.get("summary", "")}
            )
            return "Turn ended."
        if name == "send_message":
            to = args.get("to", "*")
            self.messages.append(
                {"from": actor, "to": to, "type": "message", "text": args.get("content", "")}
            )
            return f"Message sent to {to}."
        if name == "log":
            self.messages.append(
                {"from": actor, "type": "log", "text": args.get("entry", "")}
            )
            return "Logged."
        if name == "declare_outcome":
            self.outcome = args.get("outcome")
            self.winner = args.get("winner")
            self.messages.append(
                {
                    "from": actor,
                    "type": "outcome",
                    "outcome": self.outcome,
                    "winner": self.winner,
                    "reason": args.get("reason", ""),
                }
            )
            return "Outcome recorded. Game over."
        for action in self.framework.actions:
            if action.name == name:
                result = self.apply_action(action, actor, args or {})
                self.messages.append(
                    {"from": actor, "type": "action", "text": f"{name}({json.dumps(args or {})})"}
                )
                return json.dumps(result, indent=2)
        raise ValueError(f"unknown tool: {name}")


def _fmt_msg(m: dict[str, Any]) -> str:
    frm = m.get("from", "?")
    t = m.get("type")
    if t == "message":
        return f"[{frm} → {m.get('to', 'all')}] {m.get('text', '')}"
    if t == "action":
        return f"[{frm} does] {m.get('text', '')}"
    if t == "outcome":
        return f"[{frm} declares] {m.get('outcome', '')}: {m.get('reason', '')}"
    if t == "say":
        return f"[{frm} says] {m.get('text', '')}"
    if t == "turn_end":
        return f"[{frm} ends turn{f': ' + m['text'] if m.get('text') else ''}]"
    if t == "log":
        return f"[{frm} logs] {m.get('text', '')}"
    return f"[{frm}] {m.get('text', '')}"


def run_game(engine: Engine, provider: Any, cfg: ProjectConfig) -> None:
    fw = engine.framework
    print(f"\n=== {fw.name} ===")
    if fw.description:
        print(fw.description)
    print(f"Objective: {fw.objective or '(none)'}")
    print(f"Agents: {', '.join(engine.agents)}")
    print(f"Order: {fw.turn.order} · max_turns: {fw.turn.max_turns}\n")

    while engine.turn < fw.turn.max_turns and engine.outcome is None:
        engine.turn += 1
        print(f"--- Turn {engine.turn} ---")
        if fw.turn.order == "simultaneous":
            actors = list(engine.agents)
        else:
            actors = [engine.agents[(engine.turn - 1) % len(engine.agents)]]

        for name in actors:
            agent = next((a for a in cfg.agents if a.name == name), None)
            if agent is None:
                continue
            print(f"[{name}] acting...", flush=True)
            run_agent(provider, agent, engine)
            if engine.outcome:
                break

    _print_summary(engine)
    _persist(engine, cfg)


def _print_summary(engine: Engine) -> None:
    print("\n=== Game over ===")
    if engine.outcome:
        print(f"Outcome: {engine.outcome} (winner: {engine.winner or '?'})")
    else:
        print("Turn limit reached — no outcome declared.")
    print("\nFinal state:")
    print(json.dumps(engine.public_state(), indent=2))
    if engine.messages:
        print("\nTranscript:")
        for m in engine.messages:
            print(f"  {_fmt_msg(m)}")


def _persist(engine: Engine, cfg: ProjectConfig) -> None:
    base = cfg.path.parent if cfg.path else Path(".")
    try:
        (base / cfg.state_path).write_text(
            json.dumps(engine.public_state(), indent=2) + "\n", encoding="utf-8"
        )
        (base / "transcript.json").write_text(
            json.dumps(engine.messages, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nSaved state → {cfg.state_path}, transcript → transcript.json")
    except OSError as e:
        print(f"\n(warning: could not persist state: {e})")
