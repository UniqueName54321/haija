"""Agent runner: builds prompts and drives the tool-calling loop.

Every step is recorded to the engine's ``run_log`` (assistant text, reasoning,
tool calls, tool results) and streamed to the engine's observer, so the whole
game can be watched live and exported afterwards.
"""

from __future__ import annotations

import json
from typing import Any

from .config import AgentSpec
from .provider import ChatProvider, ToolCall, reasoning_param
from .tools import all_tools


def build_system_prompt(framework, agent_name: str, persona: str, tone: str) -> str:
    rules = "\n".join(f"- {r}" for r in framework.rules) or "(none)"
    win = "\n".join(f"- {w}" for w in framework.win_conditions) or "(none)"
    lose = "\n".join(f"- {l}" for l in framework.lose_conditions) or "(none)"
    tone_block = f"\nGAME TONE: {tone}\n" if tone else ""
    return f"""You are {agent_name}, an agent playing a game called "{framework.name}".

{framework.description}

YOUR PERSONA: {persona}
{tone_block}
OBJECTIVE: {framework.objective}

RULES:
{rules}

WIN CONDITIONS:
{win}

LOSE CONDITIONS:
{lose}

The engine is the single source of truth for game state. You can only observe
and change the world through your tools. Never claim a state you have not read
via get_state. You may send messages to other agents with send_message to
coordinate, negotiate, or bluff. You have a private memory (remember / recall /
recall_all) that no other agent can see — use it to track plans, suspicions, or
secret information. You may form and break alliances (form_alliance /
break_alliance). Some actions may be rejected if they're illegal — the engine
enforces the rules, so read any error and try a legal move instead. Play to
win, be decisive, and call end_turn when done.

Be socially active throughout the game. On most turns, react to the table with
send_message before or after acting. Use public messages for banter and visible
negotiation, and direct messages for private deals, warnings, coordination, or
bluffs. Converse more than once when the situation changes, while still making
a decisive game move and avoiding repetitive filler.

For card games, `top_card` is the authoritative card currently in play;
`discard_pile` is a summarized object, not an ordered public history. Use
`next_actor` for the player targeted by a newly played Draw/Skip card and
`next_turn_actor` for who will act after pending skip effects. Never infer
either from array position or from the transcript."""


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


def _format_inbox(inbox: list[dict[str, Any]]) -> str:
    if not inbox:
        return ""
    lines = ["\nMessages for you since your last turn:"]
    for m in inbox:
        who = m.get("from", "?")
        dest = "you" if m.get("to") != "*" else "everyone"
        lines.append(f"- {who} → {dest}: {m.get('text', '')}")
    return "\n".join(lines)


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


def run_agent(provider: ChatProvider, agent: AgentSpec, engine, cfg) -> dict[str, Any]:
    framework = engine.framework
    action_names = {a.name for a in framework.actions}

    persona = cfg.resolve_persona(agent)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(framework, agent.name, persona, cfg.tone)}
    ]
    inbox = engine.unread_messages(agent.name)
    user = (
        f"Turn {engine.turn}.\n\nCurrent game state (authoritative):\n"
        f"{json.dumps(engine.public_state(agent.name), indent=2)}\n"
        f"{_format_inbox(inbox)}\n"
        f"\nShared transcript:\n{format_transcript(engine.messages)}\n"
        "\nSocial expectation: use send_message at least once this turn. Choose #table "
        "or a direct message based on what would be most interesting or strategically useful."
    )
    messages.append({"role": "user", "content": user})
    tools = all_tools(framework)
    repeated_call: tuple[str, str] | None = None
    repeated_count = 0
    sent_message = False

    for _ in range(engine.max_steps_per_turn):
        if engine.stopped:
            return {"content": "(stopped)"}
        stream = getattr(provider, "chat_streaming", None)
        if callable(stream):
            def on_delta(event: dict[str, Any]) -> None:
                event_type = "assistant_stream" if event["kind"] == "content" else "reasoning_stream"
                engine.emit({"type": event_type, "agent": agent.name, **event})

            resp = stream(
                messages, tools=tools, reasoning=reasoning_param(agent.thinking), observer=on_delta
            )
        else:
            resp = provider.chat(messages, tools=tools, reasoning=reasoning_param(agent.thinking))

        if engine.stopped:
            return {"content": "(stopped)"}

        engine.record(
            {
                "type": "assistant",
                "turn": engine.turn,
                "agent": agent.name,
                "content": resp.content,
                "reasoning": resp.reasoning,
            }
        )
        if resp.content:
            engine.emit({"type": "assistant", "agent": agent.name, "content": resp.content})
        if resp.reasoning:
            engine.emit({"type": "reasoning", "agent": agent.name, "reasoning": resp.reasoning})

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
            if engine.stopped:
                return {"content": "(stopped)"}
            chat_required = tc.name == "end_turn" and not sent_message
            if chat_required:
                result = json.dumps({"error": "Send a public or direct message before ending your turn."})
            else:
                try:
                    result = engine.dispatch_tool(tc.name, tc.arguments, agent.name)
                except Exception as e:  # noqa: BLE001
                    result = json.dumps({"error": str(e)})
            if tc.name == "send_message" and '"error"' not in result:
                sent_message = True
            signature = (tc.name, json.dumps(tc.arguments, sort_keys=True))
            if signature == repeated_call:
                repeated_count += 1
            else:
                repeated_call, repeated_count = signature, 1
            engine.record(
                {
                    "type": "tool_call",
                    "turn": engine.turn,
                    "agent": agent.name,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            )
            engine.record(
                {
                    "type": "tool_result",
                    "turn": engine.turn,
                    "agent": agent.name,
                    "name": tc.name,
                    "result": result,
                }
            )
            engine.emit(
                {
                    "type": "tool",
                    "agent": agent.name,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": result,
                }
            )
            if tc.name in action_names:
                engine.record(
                    {
                        "type": "state",
                        "turn": engine.turn,
                        "label": f"after {tc.name}",
                        "state": engine.public_state(),
                    }
                )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result}
            )
            if repeated_count >= 3:
                message = f"{agent.name}'s turn ended after repeating {tc.name} three times."
                engine.emit({"type": "info", "message": message})
                engine.messages.append({"from": agent.name, "type": "turn_end", "text": message})
                return {"content": "(stalled turn ended)"}
            if tc.name == "end_turn" and not chat_required:
                return {"content": "(ended turn)"}
            if engine.outcome:
                return {"outcome": engine.outcome, "winner": engine.winner}

    return {"content": "(max steps reached)"}
