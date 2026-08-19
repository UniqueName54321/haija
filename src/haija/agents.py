"""Agent runner: builds prompts and drives the tool-calling loop."""

from __future__ import annotations

import json
from typing import Any

from .config import AgentSpec
from .provider import ChatProvider, ToolCall
from .tools import all_tools


def build_system_prompt(framework, agent: AgentSpec) -> str:
    rules = "\n".join(f"- {r}" for r in framework.rules) or "(none)"
    win = "\n".join(f"- {w}" for w in framework.win_conditions) or "(none)"
    lose = "\n".join(f"- {l}" for l in framework.lose_conditions) or "(none)"
    desc = agent.description or "(no personality given)"
    return f"""You are {agent.name}, an agent playing a game called "{framework.name}".

{framework.description}

YOUR PERSONA: {desc}

OBJECTIVE: {framework.objective}

RULES:
{rules}

WIN CONDITIONS:
{win}

LOSE CONDITIONS:
{lose}

The engine is the single source of truth for game state. You can only observe
and change the world through your tools. Never claim a state you have not read
via get_state. Play to win, be decisive, and call end_turn when done."""


def _tool_calls_to_api(tool_calls: list[ToolCall]) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
        }
        for tc in tool_calls
    ]


def format_transcript(messages: list[dict[str, Any]], limit: int = 20) -> str:
    recent = messages[-limit:]
    if not recent:
        return "(no messages yet)"
    lines = []
    for m in recent:
        frm = m.get("from", "?")
        t = m.get("type")
        if t == "message":
            lines.append(f"[{frm} → {m.get('to', 'all')}] {m.get('text', '')}")
        elif t == "action":
            lines.append(f"[{frm} does] {m.get('text', '')}")
        elif t == "outcome":
            lines.append(f"[{frm} declares] {m.get('outcome', '')}: {m.get('reason', '')}")
        elif t == "say":
            lines.append(f"[{frm} says] {m.get('text', '')}")
        elif t == "turn_end":
            lines.append(f"[{frm} ends turn{f': ' + m['text'] if m.get('text') else ''}]")
        elif t == "log":
            lines.append(f"[{frm} logs] {m.get('text', '')}")
        else:
            lines.append(f"[{frm}] {m.get('text', '')}")
    return "\n".join(lines)


def run_agent(provider: ChatProvider, agent: AgentSpec, engine) -> dict[str, Any]:
    framework = engine.framework
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(framework, agent)}
    ]
    user = (
        f"Turn {engine.turn}.\n\nCurrent game state (authoritative):\n"
        f"{json.dumps(engine.public_state(), indent=2)}\n\n"
        f"Shared transcript:\n{format_transcript(engine.messages)}"
    )
    messages.append({"role": "user", "content": user})
    tools = all_tools(framework)

    for _ in range(engine.max_steps_per_turn):
        resp = provider.chat(messages, tools=tools)
        assistant: dict[str, Any] = {"role": "assistant", "content": resp.content or None}
        api_tcs = _tool_calls_to_api(resp.tool_calls)
        if api_tcs:
            assistant["tool_calls"] = api_tcs
        messages.append(assistant)

        if not resp.tool_calls:
            if resp.content:
                engine.messages.append({"from": agent.name, "type": "say", "text": resp.content})
            return {"content": resp.content or "(no response)"}

        for tc in resp.tool_calls:
            try:
                result = engine.dispatch_tool(tc.name, tc.arguments, agent.name)
            except Exception as e:  # noqa: BLE001
                result = json.dumps({"error": str(e)})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result}
            )
            if tc.name == "end_turn":
                return {"content": "(ended turn)"}
            if engine.outcome:
                return {"outcome": engine.outcome, "winner": engine.winner}

    return {"content": "(max steps reached)"}
