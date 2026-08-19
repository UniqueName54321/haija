from pathlib import Path

from haija.engine import Engine, render_export
from haija.framework import load_framework

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _example_engine():
    fw = load_framework(EXAMPLES / "tic-tac-toe" / "framework.json")
    return Engine(fw, ["Alpha", "Beta"])


def test_unread_messages_filters_and_clears():
    e = _example_engine()
    e.dispatch_tool("send_message", {"to": "Alpha", "content": "hello Alpha"}, "Beta")
    e.dispatch_tool("send_message", {"to": "*", "content": "hi all"}, "Alpha")
    inbox = e.unread_messages("Alpha")
    assert [m["text"] for m in inbox] == ["hello Alpha", "hi all"]
    # Already-read, so a second call returns nothing new.
    assert e.unread_messages("Alpha") == []


def test_unread_messages_respects_recipient():
    e = _example_engine()
    e.dispatch_tool("send_message", {"to": "Beta", "content": "only Beta"}, "Alpha")
    assert e.unread_messages("Alpha") == []


def test_export_text_contains_all_sections():
    e = _example_engine()
    e.dispatch_tool("send_message", {"to": "Beta", "content": "glhf"}, "Alpha")
    e.record({"type": "assistant", "turn": 1, "agent": "Alpha", "content": "I'll move", "reasoning": "center is strong"})
    e.record({"type": "tool_call", "turn": 1, "agent": "Alpha", "name": "place_mark", "arguments": {"cell": 4}})
    e.record({"type": "tool_result", "turn": 1, "agent": "Alpha", "name": "place_mark", "result": "ok"})
    e.record({"type": "outcome", "turn": 1, "from": "Alpha", "outcome": "win", "winner": "Alpha", "reason": "line"})

    text = e.export_text()
    assert "HAIJA RUN EXPORT" in text
    assert "FRAMEWORK" in text
    assert "TURN 1" in text
    assert "FINAL STATE" in text
    assert "I'll move" in text
    assert "center is strong" in text  # thinking captured
    assert "place_mark" in text          # tool call captured
    assert "[chat] Alpha → Beta: glhf" in text


def test_render_export_function():
    text = render_export(
        name="T", description="d", objective="o", rules=["r1"],
        win=["w1"], lose=[], agents=["A", "B"],
        run_log=[{"type": "state", "turn": 0, "label": "initial", "state": {"x": 1}}],
        final_state={"x": 2},
    )
    assert "HAIJA RUN EXPORT" in text
    assert '"x": 1' in text
    assert '"x": 2' in text
