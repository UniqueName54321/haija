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
