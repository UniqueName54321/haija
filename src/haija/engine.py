"""The engine: owns game state (the "truth") and runs the turn loop.

The engine also keeps a full ``run_log`` — every state snapshot, assistant
response, piece of reasoning, tool call, tool result, chat message, and the
final outcome — so an entire game can be exported as a human-readable ``.txt``.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agents import run_agent
from .config import ProjectConfig
from .framework import Action, Framework
from .guards import evaluate_guards
from .templating import resolve_path, resolve_value

LOG = logging.getLogger(__name__)

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
    private: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-agent hidden memory
    alliances: set[tuple[str, str]] = field(default_factory=set)  # sorted (a, b) pairs
    observer: Observer | None = field(default=None, repr=False)
    turn: int = 0
    outcome: str | None = None
    winner: str | None = None
    stopped: bool = False
    _advanced_this_turn: bool = field(default=False, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def stop(self) -> None:
        """Request a graceful stop at the next turn boundary."""
        self.stopped = True

    def __post_init__(self) -> None:
        if not self.state:
            self.state = copy.deepcopy(self.framework.initial_state)
        seed = self.state.get("random_seed")
        if not isinstance(seed, int):
            seed = random.SystemRandom().randrange(2**63)
        self.state["_random_seed"] = seed
        self._rng = random.Random(seed)
        # Inject the agent roster into the initial state so that
        # framework-generated "players" / "hands" stubs are populated.
        if "players" in self.state and not self.state["players"]:
            self.state["players"] = list(self.agents)
        if "hands" in self.state and isinstance(self.state["hands"], dict) and not self.state["hands"]:
            self.state["hands"] = {a: [] for a in self.agents}
        self._resolve_setup_state()
        if self.agents and "turn_index" in self.state:
            self.state["turn_index"] = int(self.state.get("turn_index", 0)) % len(self.agents)

    def _resolve_setup_state(self) -> None:
        """Turn conventional generated card-game setup into a playable state.

        Generated frameworks commonly describe a deck, empty per-player hands,
        and an empty discard pile while leaving ``phase`` at ``setup``. There is
        no setup turn or deal tool, so resolve that declarative state before the
        first agent is called.
        """
        if str(self.state.get("phase", "")).lower() != "setup":
            return

        hands = self.state.get("hands")
        deck = next(
            (self.state[key] for key in ("deck", "draw_pile", "drawPile") if isinstance(self.state.get(key), list)),
            None,
        )
        if not isinstance(hands, dict) or not isinstance(deck, list):
            # A setup phase without declarative setup data must not trap agents
            # in an unwinnable polling loop.
            self.state["phase"] = "playing"
            return

        for agent in self.agents:
            hands.setdefault(agent, [])

        if deck and all(isinstance(hand, list) and not hand for hand in hands.values()):
            self._rng.shuffle(deck)
            hand_size = self._starting_hand_size()
            discard = self._discard_pile()
            reserve = 1 if discard is not None else 0
            available = max(0, len(deck) - reserve)
            hand_size = min(hand_size, available // max(1, len(self.agents)))
            for _ in range(hand_size):
                for agent in self.agents:
                    hands[agent].append(deck.pop())

        discard = self._discard_pile()
        if discard is not None and not discard and deck:
            card = self._pop_initial_discard(deck)
            discard.append(card)
            for key in ("top_card", "current_card"):
                if key in self.state:
                    self.state[key] = copy.deepcopy(card)
            color, value = self._card_attributes(card)
            if color is not None:
                for key in ("current_color", "active_color"):
                    if key in self.state:
                        self.state[key] = color
            if value is not None:
                for key in ("current_number", "current_value"):
                    if key in self.state:
                        self.state[key] = value

        self.state["phase"] = "playing"

    def _discard_pile(self) -> list[Any] | None:
        for key in ("discard_pile", "discard", "played_cards", "discardPile"):
            value = self.state.get(key)
            if isinstance(value, list):
                return value
        return None

    @staticmethod
    def _card_attributes(card: Any) -> tuple[Any, Any]:
        if isinstance(card, dict):
            return card.get("color"), card.get("number", card.get("value"))
        if isinstance(card, str) and len(card) >= 2 and card[0].upper() in "RGBY":
            value = card[1:]
            return card[0].upper(), int(value) if value.isdigit() else value
        return None, None

    @classmethod
    def _pop_initial_discard(cls, deck: list[Any]) -> Any:
        # Prefer a numbered/ordinary colored card so play starts with an
        # unambiguous color and value instead of an unresolved Wild/action.
        for index in range(len(deck) - 1, -1, -1):
            color, value = cls._card_attributes(deck[index])
            if color is not None and (isinstance(value, (int, float)) or str(value).isdigit()):
                return deck.pop(index)
        return deck.pop()

    def _starting_hand_size(self) -> int:
        for key in ("initial_hand_size", "starting_hand_size", "cards_per_player", "hand_size"):
            value = self.state.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        text = " ".join(
            [self.framework.description, *self.framework.rules]
        )
        match = re.search(
            r"(?:starts?|begins?|dealt|deals?)\s+(?:with\s+)?(\d+)\s+cards?",
            text,
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else 7

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
    def public_state(self, viewer: str | None = None) -> dict[str, Any]:
        """Return full state internally, or a private-safe view for an agent."""
        view = copy.deepcopy(self.state)
        if viewer is None:
            return view
        hands = view.get("hands")
        if isinstance(hands, dict):
            for name, hand in list(hands.items()):
                if name != viewer and isinstance(hand, list):
                    hands[name] = {"count": len(hand), "hidden": True}
        for key in ("deck", "draw_pile", "drawPile"):
            if isinstance(view.get(key), list):
                view[key] = {"count": len(view[key]), "hidden": True}
        return view

    def get_state(self, paths: list[str] | None = None, viewer: str | None = None) -> dict[str, Any]:
        full = self.public_state(viewer)
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

    def current_actor(self) -> str:
        if not self.agents:
            raise ValueError("game has no agents")
        index = int(self.state.get("turn_index", 0)) % len(self.agents)
        self.state["turn_index"] = index
        return self.agents[index]

    def next_actor(self, steps: int = 1) -> str:
        direction = -1 if int(self.state.get("direction", 1)) < 0 else 1
        index = int(self.state.get("turn_index", 0))
        return self.agents[(index + direction * steps) % len(self.agents)]

    def finish_turn(self) -> None:
        """Advance the authoritative state scheduler exactly once."""
        if not self.agents or self._advanced_this_turn:
            self._advanced_this_turn = False
            return
        skip = int(self.state.pop("skip_count", 0) or 0)
        if self.state.pop("skip_next", False):
            skip = max(skip, 1)
        direction = -1 if int(self.state.get("direction", 1)) < 0 else 1
        index = int(self.state.get("turn_index", 0))
        self.state["turn_index"] = (index + direction * (1 + skip)) % len(self.agents)

    # ---- effect application ----------------------------------------------
    def _ctx(self, actor: str, params: dict[str, Any]) -> dict[str, Any]:
        mark = actor
        marks = self.state.get("marks")
        if isinstance(marks, dict) and actor in marks:
            mark = marks[actor]
        ctx = {
            "actor": actor,
            "mark": mark,
            "params": params,
            "state": self.state,
            "turn": self.turn,
            "now": int(time.time()),
        }
        if self.agents:
            ctx["next_actor"] = self.next_actor()
        return ctx

    def apply_action(
        self, action: Action, actor: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        param_error = self._validate_params(action, params)
        if param_error:
            return {"ok": False, "error": f"invalid parameters: {param_error}"}
        ctx = self._ctx(actor, params)
        if action.guard:
            passed, reason = evaluate_guards(action.guard, ctx)
            if not passed:
                LOG.warning("guard rejected %s.%s(%s): %s", actor, action.name, params, reason)
                return {"ok": False, "error": f"illegal move: {reason}"}
        snapshot = copy.deepcopy(self.state)
        advanced_before = self._advanced_this_turn
        results = []
        try:
            for eff in action.effects:
                results.append(self._apply_one(eff, ctx))
            self._resolve_conventional_card_effects(action, actor, params)
        except Exception as exc:  # action effects are an atomic transaction
            self.state = snapshot
            self._advanced_this_turn = advanced_before
            LOG.warning("action failed %s.%s(%s): %s", actor, action.name, params, exc)
            return {"ok": False, "error": f"action failed: {exc}"}
        self.history.append(
            {
                "turn": self.turn,
                "actor": actor,
                "action": action.name,
                "params": params,
                "effects": results,
            }
        )
        self._check_judge(actor, params)
        touched_hand = any(str(e.get("path", "")).startswith("hands.") for e in action.effects)
        if touched_hand:
            self._check_empty_hand_win(actor)
        return {"ok": True, "effects": results, "state": self.public_state(actor)}

    @staticmethod
    def _validate_params(action: Action, params: dict[str, Any]) -> str | None:
        schema = action.parameters or {}
        for name in schema.get("required", []):
            if name not in params:
                return f"missing required parameter '{name}'"
        types = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "array": list, "object": dict,
        }
        for name, value in params.items():
            prop = schema.get("properties", {}).get(name, {})
            expected = types.get(prop.get("type"))
            if expected and (isinstance(value, bool) and prop.get("type") in ("integer", "number") or not isinstance(value, expected)):
                return f"parameter '{name}' must be {prop.get('type')}"
            if "enum" in prop and value not in prop["enum"]:
                return f"parameter '{name}' must be one of {prop['enum']}"
        return None

    def _check_empty_hand_win(self, actor: str) -> None:
        if self.outcome:
            return
        hands = self.state.get("hands")
        if isinstance(hands, dict) and actor in hands and hands[actor] == []:
            self._set_outcome(actor, "win", actor, "engine: hand is empty")

    def _check_judge(self, actor: str, params: dict[str, Any]) -> None:
        """Evaluate the deterministic judge; declare an outcome if a guard passes."""
        judge = self.framework.judge
        if not judge.active or self.outcome:
            return
        ctx = self._ctx(actor, params)
        if judge.win and evaluate_guards(judge.win, ctx)[0]:
            self._set_outcome(actor, "win", actor, "judge: win condition met")
        elif judge.lose and evaluate_guards(judge.lose, ctx)[0]:
            self._set_outcome(actor, "lose", None, "judge: lose condition met")
        elif judge.draw and evaluate_guards(judge.draw, ctx)[0]:
            self._set_outcome(actor, "draw", None, "judge: draw condition met")

    def _set_outcome(self, actor: str, outcome: str, winner: str | None, reason: str) -> None:
        self.outcome = outcome
        self.winner = winner
        payload = {"from": actor, "type": "outcome", "outcome": outcome, "winner": winner, "reason": reason}
        self.messages.append(payload)
        self.record({"type": "outcome", "turn": self.turn, **payload})
        self.emit({"type": "outcome", "from": actor, "outcome": outcome, "winner": winner, "reason": reason})

    @staticmethod
    def _alliance_pair(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))

    def _apply_one(self, eff: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        if self.agents:
            ctx["next_actor"] = self.next_actor()
        op = eff.get("op", "set")
        path = eff.get("path", "")
        raw = eff.get("value")
        segs = resolve_path(path, ctx)
        value = resolve_value(raw, ctx)

        if op == "if":
            passed, _ = evaluate_guards(eff.get("guard", []), ctx)
            branch = eff.get("effects" if passed else "else", [])
            applied = [self._apply_one(item, ctx) for item in branch]
            return {"op": "if", "passed": passed, "effects": applied}

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
        if op == "shuffle":
            target = self._navigate(segs)
            self._rng.shuffle(target)
            return {"op": "shuffle", "path": path}
        if op in ("draw", "move"):
            source_path = eff.get("from", "deck")
            count = int(resolve_value(eff.get("count", 1), ctx))
            if op == "draw" and source_path in ("deck", "draw_pile", "drawPile") and len(segs) == 2 and segs[0] == "hands":
                drawn = self._draw_cards(segs[1], count)
                return {"op": op, "from": source_path, "path": path, "count": drawn}
            source = self._navigate(resolve_path(source_path, ctx))
            target = self._navigate(segs)
            moved = []
            for _ in range(min(count, len(source))):
                moved.append(source.pop())
            target.extend(moved)
            return {"op": op, "from": source_path, "path": path, "count": len(moved)}
        if op == "reverse_direction":
            self.state["direction"] = -int(self.state.get("direction", 1) or 1)
            return {"op": op, "direction": self.state["direction"]}
        if op == "skip_next":
            self.state["skip_count"] = max(int(self.state.get("skip_count", 0)), int(value or 1))
            return {"op": op, "count": self.state["skip_count"]}
        if op == "advance_turn":
            steps = int(value or 1)
            direction = -1 if int(self.state.get("direction", 1)) < 0 else 1
            index = int(self.state.get("turn_index", 0))
            self.state["turn_index"] = (index + direction * steps) % len(self.agents)
            self._advanced_this_turn = True
            return {"op": op, "turn_index": self.state["turn_index"]}
        raise ValueError(f"unknown effect op: {op}")

    def _resolve_conventional_card_effects(self, action: Action, actor: str, params: dict[str, Any]) -> None:
        """Compatibility for generated card frameworks predating turn effects."""
        if not isinstance(self.state.get("hands"), dict) or self._discard_pile() is None:
            return
        card = params.get("card")
        if not isinstance(card, str):
            return
        declared = {e.get("op") for e in action.effects}
        lower = card.lower()
        if "reverse" in lower and "reverse_direction" not in declared:
            self.state["direction"] = -int(self.state.get("direction", 1) or 1)
        draw_count = 4 if ("wild4" in lower or "draw4" in lower or "+4" in lower) else 2 if ("draw2" in lower or "+2" in lower) else 0
        skip = "skip" in lower or draw_count > 0
        if draw_count and "draw" not in declared:
            self._draw_cards(self.next_actor(), draw_count)
        if skip and "skip_next" not in declared:
            self.state["skip_count"] = max(int(self.state.get("skip_count", 0)), 1)

    def _draw_cards(self, player: str, count: int) -> int:
        hands = self.state.get("hands", {})
        deck = next((self.state[k] for k in ("deck", "draw_pile", "drawPile") if isinstance(self.state.get(k), list)), [])
        hand = hands.get(player) if isinstance(hands, dict) else None
        if not isinstance(hand, list):
            return 0
        drawn = 0
        for _ in range(count):
            if not deck:
                discard = self._discard_pile()
                if discard and len(discard) > 1:
                    top = discard.pop()
                    deck.extend(discard)
                    discard.clear()
                    discard.append(top)
                    self._rng.shuffle(deck)
            if not deck:
                break
            hand.append(deck.pop())
            drawn += 1
        return drawn

    def _navigate(self, segs: list[str]) -> Any:
        cur: Any = self.state
        for seg in segs:
            if isinstance(cur, list):
                try:
                    key = int(seg)
                except (TypeError, ValueError):
                    key = cur.index(seg)
                cur = cur[key]
            else:
                cur = cur[seg]
        return cur

    def _parent_key(self, segs: list[str]) -> tuple[Any, Any]:
        parent = self._navigate(segs[:-1])
        key: Any = segs[-1]
        if isinstance(parent, list):
            try:
                key = int(key)
            except (TypeError, ValueError):
                key = parent.index(key)
        return parent, key

    # ---- tool dispatch ----------------------------------------------------
    def dispatch_tool(self, name: str, args: dict[str, Any], actor: str) -> str:
        if name == "get_state":
            return json.dumps(self.get_state(args.get("paths"), actor), indent=2)
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
            if self.framework.judge.active:
                return json.dumps(
                    {"error": "the engine determines the outcome automatically for this game"}
                )
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
        if name == "remember":
            key = str(args.get("key", ""))
            value = args.get("value")
            self.private.setdefault(actor, {})[key] = value
            self.record({"type": "remember", "turn": self.turn, "agent": actor, "key": key, "value": value})
            self.emit({"type": "remember", "agent": actor, "key": key})
            return "Remembered."
        if name == "recall":
            key = str(args.get("key", ""))
            return json.dumps(self.private.get(actor, {}).get(key), indent=2)
        if name == "recall_all":
            return json.dumps(self.private.get(actor, {}), indent=2)
        if name == "form_alliance":
            other = str(args.get("with_agent", ""))
            if not other or other == actor:
                return json.dumps({"error": "provide another agent's name"})
            self.alliances.add(self._alliance_pair(actor, other))
            self.record({"type": "alliance", "turn": self.turn, "from": actor, "with": other, "action": "form"})
            self.emit({"type": "alliance", "from": actor, "with": other, "action": "form"})
            return f"Alliance formed with {other}."
        if name == "break_alliance":
            other = str(args.get("with_agent", ""))
            pair = self._alliance_pair(actor, other)
            if pair in self.alliances:
                self.alliances.discard(pair)
                self.record({"type": "alliance", "turn": self.turn, "from": actor, "with": other, "action": "break"})
                self.emit({"type": "alliance", "from": actor, "with": other, "action": "break"})
                return f"Alliance broken with {other}."
            return f"No alliance with {other}."
        if name == "list_alliances":
            return json.dumps(sorted([list(p) for p in self.alliances]), indent=2)
        if name == "my_allies":
            allies = [
                b if a == actor else a
                for a, b in self.alliances
                if actor in (a, b)
            ]
            return json.dumps(sorted(allies), indent=2)
        for action in self.framework.actions:
            if action.name == name:
                result = self.apply_action(action, actor, args or {})
                if result.get("ok"):
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
        elif et == "remember":
            L.append(f"\n[private] {entry['agent']} remembers '{entry['key']}'")
        elif et == "alliance":
            verb = "forms alliance with" if entry.get("action") == "form" else "breaks alliance with"
            L.append(f"\n[alliance] {entry['from']} {verb} {entry['with']}")
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
        if engine.stopped:
            obs({"type": "info", "message": "Run stopped by user."})
            break
        engine.turn += 1
        if fw.turn.order == "simultaneous":
            actors = list(engine.agents)
        else:
            actors = [engine.current_actor()]

        for name in actors:
            agent = next((a for a in cfg.agents if a.name == name), None)
            if agent is None:
                if fw.turn.order != "simultaneous":
                    engine.finish_turn()
                continue
            obs({"type": "turn_start", "turn": engine.turn, "agent": name})
            run_agent(provider, agent, engine, cfg)
            if engine.outcome or engine.stopped:
                break
            if fw.turn.order != "simultaneous":
                engine.finish_turn()

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
    elif t == "remember":
        print(f"[{event['agent']} remembers] {event['key']}")
    elif t == "alliance":
        verb = "forms alliance with" if event.get("action") == "form" else "breaks alliance with"
        print(f"[alliance] {event['from']} {verb} {event['with']}")
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
