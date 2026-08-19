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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnConfig":
        return cls(
            order=d.get("order", "round_robin"),
            max_turns=int(d.get("max_turns", 20)),
        )


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
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Framework":
        actions = [
            Action(
                name=a["name"],
                description=a.get("description", ""),
                parameters=a.get("parameters", {"type": "object", "properties": {}}),
                effects=list(a.get("effects", [])),
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
                }
                for a in self.actions
            ],
            "turn": {"order": self.turn.order, "max_turns": self.turn.max_turns},
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
      "effects": [ { "op": "set", "path": "dot.path.to.state.key", "value": "..." } ]
    }
  ],
  "turn": { "order": "round_robin", "max_turns": 20 }
}

Effect rules:
- "op" is one of: set (write a value), incr (add a number), append (add to a
  list), remove (delete a key/index).
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
      "effects": [ { "op": "set", "path": "dot.path.to.state.key", "value": "..." } ]
    }
  ],
  "turn": { "order": "round_robin", "max_turns": 20 }
}

Effect rules:
- "op" is one of: set (write a value), incr (add a number), append (add to a
  list), remove (delete a key/index).
- "path" is a dot-separated path into the state, e.g. "board.0" or
  "players.Alice.hp". List indices are integers.
- Path segments and values may contain templates: {{actor}}, {{mark}},
  {{params.x}}, {{state.some.key}}, {{turn}}, {{now}}.
- Every action agents need MUST be listed under "actions"; agents change state
  only through actions. The engine is the single source of truth.

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
        "turn": {"order": order, "max_turns": max_turns},
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
