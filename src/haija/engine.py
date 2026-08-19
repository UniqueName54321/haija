"""The engine: owns game state (the "truth") and runs the turn loop.

The engine also keeps a full ``run_log`` — every state snapshot, assistant
response, piece of reasoning, tool call, tool result, chat message, and the
final outcome — so an entire game can be exported as a human-readable ``.txt``.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agents import run_agent
from .config import ProjectConfig
from .framework import Action, Framework
from .templating import resolve_path, resolve_value

Observer = Callable[[dict[str, Any]], None]


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


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


@dataclass
class Engine:
    framework: Framework
    agents: list[str]
    max_steps_per_turn: int = 16
    state: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    run_log: list[dict[str, Any]] = field(default_factory=list)
    last_seen: dict[str, int] = field(default_factory=dict)
    observer: Observer | None = field(default=None, repr=False)
    turn: int = 0
    outcome: str | None = None
    winner: str | None = None

    def __post_init__(self) -> None:
        if not self.state:
            self.state = copy.deepcopy(self.framework.initial_state)

    # ---- recording / streaming -------------------------------------------
    def record(self, entry: dict[str, Any]) -> None:
        self.run_log.append(entry)

    def emit(self, event: dict[str, Any]) -> None:
        if self.observer:
            self.observer(event)

    def unread_messages(self, agent: str) -> list[dict[str, Any]]:
        """Chat messages addressed to ``agent`` since their last turn."""
        start = self.last_seen.get(agent, 0)
        chat = [
            m
            for m in self.messages[start:]
            if m.get("type") == "message" and m.get("to") in ("*", agent)
        ]
        self.last_seen[agent] = len(self.messages)
        return chat

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
            text = args.get("content", "")
            msg = {"from": actor, "to": to, "type": "message", "text": text}
            self.messages.append(msg)
            self.record({"type": "message", "turn": self.turn, **msg})
            self.emit({"type": "message", "from": actor, "to": to, "text": text})
            return f"Message sent to {to}."
        if name == "log":
            self.messages.append(
                {"from": actor, "type": "log", "text": args.get("entry", "")}
            )
            self.record(
                {"type": "log", "turn": self.turn, "from": actor, "text": args.get("entry", "")}
            )
            self.emit({"type": "log", "from": actor, "text": args.get("entry", "")})
            return "Logged."
        if name == "declare_outcome":
            outcome = args.get("outcome")
            winner = args.get("winner")
            reason = args.get("reason", "")
            self.outcome = outcome
            self.winner = winner
            self.messages.append(
                {
                    "from": actor,
                    "type": "outcome",
                    "outcome": outcome,
                    "winner": winner,
                    "reason": reason,
                }
            )
            self.record(
                {
                    "type": "outcome",
                    "turn": self.turn,
                    "from": actor,
                    "outcome": outcome,
                    "winner": winner,
                    "reason": reason,
                }
            )
            self.emit(
                {
                    "type": "outcome",
                    "from": actor,
                    "outcome": outcome,
                    "winner": winner,
                    "reason": reason,
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

    # ---- export -----------------------------------------------------------
    def export_text(self) -> str:
        return render_export(
            name=self.framework.name,
            description=self.framework.description,
            objective=self.framework.objective,
            rules=self.framework.rules,
            win=self.framework.win_conditions,
            lose=self.framework.lose_conditions,
            agents=self.agents,
            run_log=self.run_log,
            final_state=self.public_state(),
        )


def render_export(
    name: str,
    description: str,
    objective: str,
    rules: list[str],
    win: list[str],
    lose: list[str],
    agents: list[str],
    run_log: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> str:
    """Render a full run log as a human-readable ``.txt`` document."""
    L: list[str] = []
    L.append("=" * 72)
    L.append("HAIJA RUN EXPORT")
    L.append("=" * 72)
    L.append(f"Game:      {name}")
    L.append(f"Exported:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Agents:    {', '.join(agents)}")
    L.append("")
    L.append("=" * 72)
    L.append("FRAMEWORK")
    L.append("=" * 72)
    if description:
        L.append(description)
    if objective:
        L.append(f"\nObjective: {objective}")
    if rules:
        L.append("\nRules:")
        L += [f"  - {r}" for r in rules]
    if win:
        L.append("\nWin conditions:")
        L += [f"  - {w}" for w in win]
    if lose:
        L.append("\nLose conditions:")
        L += [f"  - {l}" for l in lose]

    cur_turn: int | None = None
    for entry in run_log:
        turn = entry.get("turn")
        if turn != cur_turn:
            cur_turn = turn
            L.append("")
            L.append("=" * 72)
            L.append("SETUP" if not turn else f"TURN {turn}")
            L.append("=" * 72)

        et = entry["type"]
        if et == "state":
            L.append(f"\n[state · {entry.get('label', '')}]")
            L.append(json.dumps(entry["state"], indent=2))
        elif et == "assistant":
            L.append(f"\n[{entry['agent']} → response]")
            L.append(entry.get("content") or "(no text)")
            if entry.get("reasoning"):
                L.append(f"\n[{entry['agent']} → thinking]")
                L.append(_indent(entry["reasoning"], "  "))
        elif et == "tool_call":
            L.append(
                f"\n[{entry['agent']} → tool] {entry['name']}"
                f"({json.dumps(entry.get('arguments') or {})})"
            )
        elif et == "tool_result":
            L.append(f"  → {entry['name']} result:")
            L.append(_indent(str(entry.get("result", "")), "      "))
        elif et == "message":
            L.append(f"\n[chat] {entry['from']} → {entry['to']}: {entry['text']}")
        elif et == "log":
            L.append(f"\n[log] {entry['from']}: {entry['text']}")
        elif et == "outcome":
            L.append(
                f"\n[outcome] {entry['from']} declares {entry['outcome']} "
                f"(winner {entry.get('winner') or '?'}): {entry.get('reason', '')}"
            )

    L.append("")
    L.append("=" * 72)
    L.append("FINAL STATE")
    L.append("=" * 72)
    L.append(json.dumps(final_state, indent=2))
    return "\n".join(L) + "\n"


def run_game(
    engine: Engine, provider: Any, cfg: ProjectConfig, observer: Observer | None = None
) -> str:
    obs = observer or print_observer
    engine.observer = obs
    fw = engine.framework

    engine.record({"type": "state", "turn": 0, "label": "initial", "state": engine.public_state()})
    obs(
        {
            "type": "game_start",
            "name": fw.name,
            "description": fw.description,
            "objective": fw.objective,
            "agents": engine.agents,
            "order": fw.turn.order,
            "max_turns": fw.turn.max_turns,
        }
    )

    while engine.turn < fw.turn.max_turns and engine.outcome is None:
        engine.turn += 1
        if fw.turn.order == "simultaneous":
            actors = list(engine.agents)
        else:
            actors = [engine.agents[(engine.turn - 1) % len(engine.agents)]]

        for name in actors:
            agent = next((a for a in cfg.agents if a.name == name), None)
            if agent is None:
                continue
            obs({"type": "turn_start", "turn": engine.turn, "agent": name})
            run_agent(provider, agent, engine, cfg)
            if engine.outcome:
                break

    obs(
        {
            "type": "game_over",
            "outcome": engine.outcome,
            "winner": engine.winner,
            "final_state": engine.public_state(),
        }
    )
    return _persist(engine, cfg)


def _persist(engine: Engine, cfg: ProjectConfig) -> str:
    base = cfg.path.parent if cfg.path else Path(".")
    try:
        (base / cfg.state_path).write_text(
            json.dumps(engine.public_state(), indent=2) + "\n", encoding="utf-8"
        )
        (base / "transcript.json").write_text(
            json.dumps(engine.messages, indent=2) + "\n", encoding="utf-8"
        )
        (base / "run.json").write_text(
            json.dumps(engine.run_log, indent=2) + "\n", encoding="utf-8"
        )
        return f"Saved state → {cfg.state_path}, run log → run.json, transcript → transcript.json"
    except OSError as e:
        return f"(warning: could not persist state: {e})"


def print_observer(event: dict[str, Any]) -> None:
    """Default observer that streams a game to the terminal."""
    t = event.get("type")
    if t == "game_start":
        print(f"\n=== {event['name']} ===")
        if event.get("description"):
            print(event["description"])
        print(f"Objective: {event.get('objective') or '(none)'}")
        print(f"Agents: {', '.join(event['agents'])}")
        print(f"Order: {event['order']} · max_turns: {event['max_turns']}\n")
    elif t == "turn_start":
        print(f"--- Turn {event['turn']} — {event['agent']} ---", flush=True)
    elif t == "assistant":
        if event.get("content"):
            print(f"[{event['agent']} says] {event['content']}")
    elif t == "reasoning":
        if event.get("reasoning"):
            print(f"[{event['agent']} thinks] {event['reasoning']}")
    elif t == "tool":
        print(f"[{event['agent']} → {event['name']}({json.dumps(event.get('arguments') or {})})]")
    elif t == "message":
        print(f"[{event['from']} → {event['to']}] {event['text']}")
    elif t == "log":
        print(f"[{event['from']} logs] {event['text']}")
    elif t == "outcome":
        print(f"[{event['from']} declares] {event['outcome']} (winner {event.get('winner') or '?'}): {event.get('reason', '')}")
    elif t == "game_over":
        print("\n=== Game over ===")
        if event.get("outcome"):
            print(f"Outcome: {event['outcome']} (winner: {event.get('winner') or '?'})")
        else:
            print("Turn limit reached — no outcome declared.")
        print("Final state:")
        print(json.dumps(event.get("final_state", {}), indent=2))
