"""Built-in tool schemas + the merged tool list exposed to agents."""

from __future__ import annotations

from typing import Any

from .framework import Framework


def builtin_tools() -> list[dict[str, Any]]:
    """Tools every agent always has, regardless of the framework."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_state",
                "description": (
                    "Read the current game state — the authoritative truth. "
                    "Optionally narrow to specific dot-paths (e.g. ['board']). "
                    "Always read before claiming anything about the world."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dot-paths to read. Omit for the full state.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": (
                    "Send a message to another agent or to all agents. Use it to "
                    "coordinate, negotiate, bluff, or trash-talk."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Agent name, or '*' for all."},
                        "content": {"type": "string", "description": "Message text."},
                    },
                    "required": ["to", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "log",
                "description": "Write a note to the shared game transcript (visible to all).",
                "parameters": {
                    "type": "object",
                    "properties": {"entry": {"type": "string"}},
                    "required": ["entry"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "end_turn",
                "description": "End your turn. Optionally add a short note about what you did.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "declare_outcome",
                "description": "Declare the game over. The engine records it and ends the game.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "enum": ["win", "lose", "draw"]},
                        "winner": {"type": "string", "description": "Winning agent name (for 'win')."},
                        "reason": {"type": "string", "description": "Why the game is over."},
                    },
                    "required": ["outcome", "reason"],
                },
            },
        },
    ]


def all_tools(framework: Framework) -> list[dict[str, Any]]:
    """Built-in tools plus every framework-defined action."""
    tools = builtin_tools()
    for action in framework.actions:
        tools.append(action.to_tool())
    return tools
