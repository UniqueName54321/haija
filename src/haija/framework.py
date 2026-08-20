"""The framework file: Haija's single source of truth for a game.

A *framework* fully defines a playable game:

- ``description`` / ``objective`` / ``rules`` / ``win_conditions`` /
  ``lose_conditions`` — the human- and agent-readable "truth" of the game.
- ``initial_state`` — the starting world state (arbitrary JSON).
- ``actions`` — the tools agents may call, each with a JSON-schema parameter
  block and a list of declarative ``effects`` that mutate state.

The engine owns the world state. Agents can only observe it through ``get_state``
and change it through actions — so the framework + engine, not the model, decide
what is actually true.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"


@dataclass
class Action:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    effects: list[dict[str, Any]] = field(default_factory=list)
    guard: list[dict[str, Any]] = field(default_factory=list)  # preconditions (AND)
    turn_cost: dict[str, int] = field(default_factory=dict)

    def to_tool(self) -> dict[str, Any]:
        params = self.parameters or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


@dataclass
class TurnConfig:
    order: str = "round_robin"  # "round_robin" | "simultaneous"
    max_turns: int = 20
    action_limits: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnConfig":
        return cls(
            order=d.get("order", "round_robin"),
            max_turns=int(d.get("max_turns", 20)),
            action_limits=_int_map(d.get("action_limits")),
        )


@dataclass
class Judge:
    """Deterministic win/lose/draw conditions the engine evaluates itself."""

    win: list[dict[str, Any]] = field(default_factory=list)
    lose: list[dict[str, Any]] = field(default_factory=list)
    draw: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.win or self.lose or self.draw)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Judge":
        d = d or {}
        return cls(
            win=_as_guard_list(d.get("win")),
            lose=_as_guard_list(d.get("lose")),
            draw=_as_guard_list(d.get("draw")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"win": self.win, "lose": self.lose, "draw": self.draw}


@dataclass
class Framework:
    name: str
    description: str = ""
    objective: str = ""
    rules: list[str] = field(default_factory=list)
    win_conditions: list[str] = field(default_factory=list)
    lose_conditions: list[str] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)
    turn: TurnConfig = field(default_factory=TurnConfig)
    judge: Judge = field(default_factory=Judge)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Framework":
        actions = [
            Action(
                name=a["name"],
                description=a.get("description", ""),
                parameters=a.get("parameters", {"type": "object", "properties": {}}),
                effects=list(a.get("effects", [])),
                guard=_as_guard_list(a.get("guard")),
                turn_cost=_int_map(a.get("turn_cost")),
            )
            for a in d.get("actions", [])
        ]
        return cls(
            name=d.get("name", "Untitled Game"),
            description=d.get("description", ""),
            objective=d.get("objective", ""),
            rules=list(d.get("rules", [])),
            win_conditions=list(d.get("win_conditions", [])),
            lose_conditions=list(d.get("lose_conditions", [])),
            initial_state=d.get("initial_state", {}),
            actions=actions,
            turn=TurnConfig.from_dict(d.get("turn", {})),
            judge=Judge.from_dict(d.get("judge")),
            raw=d,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "objective": self.objective,
            "rules": self.rules,
            "win_conditions": self.win_conditions,
            "lose_conditions": self.lose_conditions,
            "initial_state": self.initial_state,
            "actions": [
                {
                    "name": a.name,
                    "description": a.description,
                    "parameters": a.parameters,
                    "effects": a.effects,
                    "guard": a.guard,
                    "turn_cost": a.turn_cost,
                }
                for a in self.actions
            ],
            "turn": {
                "order": self.turn.order,
                "max_turns": self.turn.max_turns,
                "action_limits": self.turn.action_limits,
            },
            "judge": self.judge.to_dict(),
        }


def load_framework(path: Path) -> Framework:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Framework.from_dict(data)


def framework_schema_instructions() -> str:
    """Schema description used when generating frameworks via an LLM."""
    return """A Haija framework is a single JSON object with this exact shape:

{
  "schema_version": "1",
  "name": "short game name",
  "description": "the world/premise, a few sentences",
  "objective": "what the players are trying to achieve",
  "rules": ["hard rules of the game", "..."],
  "win_conditions": ["conditions that mean a player wins"],
  "lose_conditions": ["conditions that mean a player loses"],
  "initial_state": { "...": "the starting world state (JSON, may be nested)" },
  "actions": [
    {
      "name": "action_name",
      "description": "what this action does, written for the agent",
      "parameters": {
        "type": "object",
        "properties": { "x": { "type": "string", "description": "..." } },
        "required": ["x"]
      },
      "effects": [ { "op": "set", "path": "dot.path.to.state.key", "value": "..." } ],
      "turn_cost": { "standard_action": 1 }
    }
  ],
  "turn": { "order": "round_robin", "max_turns": 20,
            "action_limits": { "standard_action": 1 } }
}

Effect rules:
- "op" may be set, incr, append, remove, shuffle, random_int,
  random_choice, draw/move, if,
  reverse_direction, skip_next, or advance_turn.
- draw/move uses "from", "path", and optional "count". Example:
  {"op":"draw","from":"deck","path":"hands.{{next_actor}}","count":4}.
- if uses "guard", "effects", and optional "else" effect lists.
- random_int uses "path", inclusive integer "min" and "max".
- random_choice uses "path" and a non-empty "choices" array.
- `turn.action_limits` declares named resources refreshed for each player turn.
  `action.turn_cost` consumes those resources only after a successful action.
- "path" is a dot-separated path into the state, e.g. "board.0" or
  "players.Alice.hp". List indices are integers.
- Path segments and values may contain templates: {{actor}} (the acting agent's
  name), {{mark}} (the agent's mark, if state.marks maps agent->mark),
  {{params.x}} (an action parameter), {{state.some.key}}, {{turn}}, {{now}}.
- Every action agents need MUST be listed under "actions"; agents can only
  change state through these actions. The engine is the single source of truth.

Return ONLY valid JSON. No markdown fences, no commentary."""


def framework_content_instructions() -> str:
    """Schema description for the *content* an LLM authors.

    The model produces only the game-design fields; Haija fills in the envelope
    (``name`` and ``schema_version``) and normalizes the result, so the model
    can't corrupt the file's shape.
    """
    return """You are filling in the game-design *content* for a Haija framework.
Return ONLY a JSON object with these fields (no markdown fences, no commentary):

{
  "description": "the world/premise, a few sentences",
  "objective": "what the players are trying to achieve",
  "rules": ["hard rules of the game", "..."],
  "win_conditions": ["conditions that mean a player wins"],
  "lose_conditions": ["conditions that mean a player loses"],
  "initial_state": { "...": "the starting world state (JSON, may be nested)" },
  "actions": [
    {
      "name": "action_name",
      "description": "what this action does, written for the agent",
      "parameters": {
        "type": "object",
        "properties": { "x": { "type": "string", "description": "..." } },
        "required": ["x"]
      },
      "effects": [ { "op": "set", "path": "dot.path.to.state.key", "value": "..." } ],
      "turn_cost": { "standard_action": 1 },
      "guard": [ { "op": "not_exists", "path": "board.{{params.cell}}" } ]
    }
  ],
  "turn": { "order": "round_robin", "max_turns": 20,
            "action_limits": { "standard_action": 1 } },
  "judge": {
    "win":  [ { "op": "eq", "path": "winner", "value": "{{actor}}" } ],
    "lose": [],
    "draw": []
  }
}

Effect rules:
- Effect ops: set, incr, append, remove, shuffle, random_int, random_choice,
  draw, move, if,
  reverse_direction, skip_next, advance_turn.
- draw/move: {"op":"draw","from":"deck","path":"hands.{{next_actor}}","count":4}.
- if: {"op":"if","guard":[...],"effects":[...],"else":[...]}.
- Random integer: {"op":"random_int","path":"die","min":1,"max":6}.
  Bounds are inclusive. Never fake randomness by setting an empty value.
- Random selection: {"op":"random_choice","path":"weather","choices":["sun","rain"]}.
- If rules allow only one of several actions per turn, declare a shared limit,
  e.g. `turn.action_limits: {"standard_action":1}`, and give every mutually
  exclusive action `turn_cost: {"standard_action":1}`. Card play actions and
  Nomic roll/propose actions must consume a standard action. Put all consequences
  of one action (rolling, scoring, recording the result) in that action's effects.
- Every stated result and score threshold must be reachable. A d6 produces only
  1 through 6; do not create conditions for 8 or 12 unless rolling multiple dice.
- Use reverse_direction, skip_next, and draw effects for action cards; prose
  rules alone do not mutate state. The engine advances ordinary turns.
- For UNO, set `deck_type` to `"uno"`; Haija constructs/shuffles the standard
  108-card deck and deals it. Never represent an intended full deck as `[]`.
- A draw action guard must use `length_gt` on the draw pile with value 0. Do
  NOT use `not_exists`: an existing draw pile is required, not forbidden.
- Capture or append a selected list item before removing it. Conditional card
  effects must inspect the selected card as it existed before removal.
- Treat `top_card` as the authoritative current discard. Do not infer the top
  from raw discard history or describe an array-end convention to agents.
- "path" is a dot-separated path into the state, e.g. "board.0" or
  "players.Alice.hp". List indices are integers.
- Path segments and values may contain templates: {{actor}}, {{mark}},
  {{params.x}}, {{state.some.key}}, {{turn}}, {{now}}.
- Every action agents need MUST be listed under "actions"; agents change state
  only through actions. The engine is the single source of truth.
- `initial_state` MUST be immediately playable on turn 1. Never emit an
  unresolved setup/loading/dealing phase, empty required hands, or a state that
  expects an agent to initialize the game. Put all setup results directly in
  `initial_state`; the engine only injects configured agent names into empty
  `players` and `hands` mappings.

Guard rules (optional):
- An action may have a "guard" list of conditions that MUST all be true for the
  action to be legal; the engine rejects illegal moves automatically.
- The top-level "judge" lets the engine decide the outcome itself: a passing
  "win" guard means the acting agent wins, "lose" means it loses, "draw" is a
  draw. Omit "judge" (or leave its lists empty) to let agents declare outcomes.
- Guard ops: eq, neq, gt, gte, lt, lte, in, not_in, contains, not_contains,
  exists, not_exists, length_eq, length_gt/gte/lt/lte. Boolean groups may use
  {"all":[...]}, {"any":[...]}, or {"not":{...}}. Each simple
  guard is {"op": ..., "path": "...", "value": ...}, or {"not": {...}}.

Do NOT include "name" or "schema_version" — Haija supplies those."""


def _as_str(value: Any) -> str:
    return str(value) if value is not None else ""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_guard_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [g for g in value if isinstance(g, dict)]
    return []


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, amount in value.items():
        try:
            parsed = int(amount)
        except (TypeError, ValueError):
            continue
        if str(key).strip():
            result[str(key).strip()] = parsed
    return result


def normalize_content(content: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce an AI-authored game-design object into a well-formed framework
    body, filling deterministic defaults for anything missing or malformed."""
    content = content or {}
    actions: list[dict[str, Any]] = []
    for a in content.get("actions") or []:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name", "")).strip()
        if not name:
            continue
        params = a.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        effects = a.get("effects")
        if not isinstance(effects, list):
            effects = []
        actions.append(
            {
                "name": name,
                "description": _as_str(a.get("description")),
                "parameters": params,
                "effects": [e for e in effects if isinstance(e, dict)],
                "guard": _as_guard_list(a.get("guard")),
                "turn_cost": _int_map(a.get("turn_cost")),
            }
        )

    turn = content.get("turn")
    if not isinstance(turn, dict):
        turn = {}
    order = turn.get("order")
    if order not in ("round_robin", "simultaneous"):
        order = "round_robin"
    try:
        max_turns = int(turn.get("max_turns", 20))
    except (TypeError, ValueError):
        max_turns = 20

    return {
        "description": _as_str(content.get("description")),
        "objective": _as_str(content.get("objective")),
        "rules": _as_str_list(content.get("rules")),
        "win_conditions": _as_str_list(content.get("win_conditions")),
        "lose_conditions": _as_str_list(content.get("lose_conditions")),
        "initial_state": _as_dict(content.get("initial_state")),
        "actions": actions,
        "turn": {
            "order": order,
            "max_turns": max_turns,
            "action_limits": _int_map(turn.get("action_limits")),
        },
        "judge": Judge.from_dict(content.get("judge")).to_dict(),
    }


def assemble_framework(name: str, content: dict[str, Any] | None) -> Framework:
    """Assemble a full framework from a programmatic ``name`` plus AI content.

    ``name`` (the project name) is authoritative; a stray model ``name`` is only
    a fallback when no name is supplied.
    """
    body = normalize_content(content)
    fallback = str((content or {}).get("name") or "").strip()
    resolved_name = name.strip() or fallback or "Untitled Game"
    return Framework.from_dict(
        {"schema_version": SCHEMA_VERSION, "name": resolved_name, **body}
    )


def validate_framework(framework: Framework) -> tuple[list[str], list[str]]:
    """Return semantic ``(errors, warnings)`` beyond JSON shape validation."""
    errors: list[str] = []
    warnings: list[str] = []
    allowed = {
        "set", "incr", "append", "remove", "shuffle", "random_int",
        "random_choice", "draw", "move", "if", "reverse_direction",
        "skip_next", "advance_turn",
    }
    names: set[str] = set()
    guard_ops = {
        "eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "contains",
        "not_contains", "exists", "not_exists", "length_eq", "length_gt",
        "length_gte", "length_lt", "length_lte",
    }
    random_ranges: dict[str, tuple[int, int]] = {}

    def initial_value(path: str) -> Any:
        if not path or "{{" in path:
            return None
        current: Any = framework.initial_state
        try:
            for segment in path.split("."):
                current = current[int(segment)] if isinstance(current, list) else current[segment]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return current

    def effect_ops(effects: list[dict[str, Any]]) -> set[str]:
        result: set[str] = set()
        for effect in effects:
            result.add(str(effect.get("op", "set")))
            result.update(effect_ops(effect.get("effects", [])))
            result.update(effect_ops(effect.get("else", [])))
        return result

    def check_guard(guard: dict[str, Any], location: str) -> None:
        for group in ("all", "any"):
            if group in guard:
                for child in guard.get(group, []):
                    if isinstance(child, dict):
                        check_guard(child, location)
                return
        if "not" in guard:
            if isinstance(guard["not"], dict):
                check_guard(guard["not"], location)
            return
        op = guard.get("op", "eq")
        if op not in guard_ops:
            errors.append(f"{location} uses unknown guard op '{op}'")
        if not guard.get("path"):
            errors.append(f"{location} guard '{op}' requires a path")

    def check_effect(effect: dict[str, Any], action_name: str) -> None:
        op = effect.get("op", "set")
        if op not in allowed:
            errors.append(f"action '{action_name}' uses unknown effect op '{op}'")
        if op in (
            "set", "incr", "append", "remove", "shuffle", "random_int",
            "random_choice", "draw", "move",
        ) and not effect.get("path"):
            errors.append(f"action '{action_name}' effect '{op}' requires a path")
        if op in ("draw", "move") and not effect.get("from"):
            errors.append(f"action '{action_name}' effect '{op}' requires a from path")
        if op == "random_int":
            low, high = effect.get("min"), effect.get("max")
            if (
                isinstance(low, bool) or isinstance(high, bool)
                or not isinstance(low, int) or not isinstance(high, int)
            ):
                errors.append(
                    f"action '{action_name}' random_int requires integer min and max"
                )
            elif low > high:
                errors.append(f"action '{action_name}' random_int min exceeds max")
            elif effect.get("path"):
                random_ranges[str(effect["path"])] = (low, high)
        if op == "random_choice" and not (
            isinstance(effect.get("choices"), list) and effect["choices"]
        ):
            errors.append(
                f"action '{action_name}' random_choice requires a non-empty choices array"
            )
        if op == "set" and effect.get("value") == "":
            existing = initial_value(str(effect.get("path", "")))
            if isinstance(existing, (int, float)) and not isinstance(existing, bool):
                errors.append(
                    f"action '{action_name}' sets numeric state "
                    f"'{effect.get('path')}' to an empty string"
                )
        if op == "if":
            if not effect.get("guard"):
                errors.append(f"action '{action_name}' conditional effect requires a guard")
            condition = effect.get("guard", [])
            if isinstance(condition, dict):
                condition = [condition]
            for guard in condition:
                if isinstance(guard, dict):
                    check_guard(guard, f"action '{action_name}' conditional")
            for branch in (effect.get("effects", []), effect.get("else", [])):
                for child in branch:
                    if isinstance(child, dict):
                        check_effect(child, action_name)

    for action in framework.actions:
        if action.name in names:
            errors.append(f"duplicate action name '{action.name}'")
        names.add(action.name)
        if not action.effects and not action.name.lower().startswith(("say_", "pass", "end_")):
            warnings.append(f"action '{action.name}' has no effects")
        lower_name = action.name.lower()
        name_tokens = set(lower_name.replace("-", "_").split("_"))
        if name_tokens.intersection({"roll", "dice", "die"}) and not (
            effect_ops(action.effects) & {"random_int", "random_choice"}
        ):
            errors.append(
                f"action '{action.name}' describes a roll but has no random effect"
            )
        for resource, cost in action.turn_cost.items():
            if cost <= 0:
                errors.append(
                    f"action '{action.name}' turn cost '{resource}' must be positive"
                )
                continue
            limit = framework.turn.action_limits.get(resource)
            if limit is None:
                errors.append(
                    f"action '{action.name}' consumes undeclared turn resource '{resource}'"
                )
            elif cost > limit:
                errors.append(
                    f"action '{action.name}' costs {cost} '{resource}' but turn limit is {limit}"
                )
        for guard in action.guard:
            check_guard(guard, f"action '{action.name}'")
        for effect in action.effects:
            check_effect(effect, action.name)
    if not framework.actions:
        warnings.append("framework has no game actions")
    state = framework.initial_state
    if str(state.get("phase", "")).lower() == "setup":
        has_deck = state.get("deck_type") == "uno" or any(isinstance(state.get(k), list) for k in ("deck", "draw_pile", "drawPile"))
        if not isinstance(state.get("hands"), dict) or not has_deck:
            errors.append("setup phase requires hands plus a deck/draw_pile")
    if framework.turn.max_turns <= 0:
        errors.append("turn.max_turns must be positive")
    for resource, limit in framework.turn.action_limits.items():
        if limit <= 0:
            errors.append(f"turn action limit '{resource}' must be positive")

    # Rules that explicitly make generated actions mutually exclusive need a
    # shared finite resource; prose alone cannot enforce that restriction.
    for rule in framework.rules:
        lower_rule = rule.lower()
        if "either" not in lower_rule or " or " not in lower_rule:
            continue
        mentioned = [
            action for action in framework.actions
            if action.name.lower().split("_")[0] in lower_rule
        ]
        if len(mentioned) < 2:
            continue
        shared = set(mentioned[0].turn_cost)
        for action in mentioned[1:]:
            shared.intersection_update(action.turn_cost)
        shared.intersection_update(framework.turn.action_limits)
        if not shared:
            errors.append(
                "mutually exclusive actions named in rule require a shared turn_cost resource"
            )
            break

    def check_reachable_guard(guard: dict[str, Any], location: str) -> None:
        for group in ("all", "any"):
            for child in guard.get(group, []):
                if isinstance(child, dict):
                    check_reachable_guard(child, location)
        if isinstance(guard.get("not"), dict):
            check_reachable_guard(guard["not"], location)
        path = str(guard.get("path", ""))
        value = guard.get("value")
        if guard.get("op", "eq") == "eq" and path in random_ranges:
            low, high = random_ranges[path]
            if isinstance(value, (int, float)) and not low <= value <= high:
                errors.append(
                    f"{location} requires {path}={value}, outside generated range {low}..{high}"
                )

    for action in framework.actions:
        for guard in action.guard:
            check_reachable_guard(guard, f"action '{action.name}'")
        def scan_effect_guards(effects: list[dict[str, Any]]) -> None:
            for effect in effects:
                if effect.get("op") == "if":
                    guards = effect.get("guard", [])
                    if isinstance(guards, dict):
                        guards = [guards]
                    for guard in guards:
                        if isinstance(guard, dict):
                            check_reachable_guard(guard, f"action '{action.name}' conditional")
                    scan_effect_guards(effect.get("effects", []))
                    scan_effect_guards(effect.get("else", []))
        scan_effect_guards(action.effects)

    dice_ranges = [
        (path, bounds)
        for path, bounds in random_ranges.items()
        if path.lower().split(".")[-1] in {"die", "dice", "roll", "roll_result"}
    ]
    if len(dice_ranges) == 1:
        path, (low, high) = dice_ranges[0]
        roll_values = re.compile(
            r"\broll(?:s|ed|ing)?\s+(?:is\s+|of\s+)?"
            r"(\d+(?:\s*,\s*\d+)*(?:\s*,?\s*(?:or|and)\s*\d+)?)",
            re.IGNORECASE,
        )
        for rule in framework.rules:
            for match in roll_values.finditer(rule):
                for value in map(int, re.findall(r"\d+", match.group(1))):
                    if not low <= value <= high:
                        errors.append(
                            f"rule names roll {value}, outside generated range "
                            f"{path}={low}..{high}"
                        )
    for label, guards in (("judge.win", framework.judge.win), ("judge.lose", framework.judge.lose), ("judge.draw", framework.judge.draw)):
        for guard in guards:
            check_guard(guard, label)
            check_reachable_guard(guard, label)
    if not framework.judge.active and not isinstance(state.get("hands"), dict):
        warnings.append("no deterministic judge; agents must declare the outcome")
    return errors, warnings
