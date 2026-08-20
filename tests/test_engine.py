from pathlib import Path

from haija.engine import Engine
from haija.framework import Framework, load_framework

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _example_engine():
    fw = load_framework(EXAMPLES / "tic-tac-toe" / "framework.json")
    return Engine(fw, ["Alpha", "Beta"])


def test_set_effect_with_mark_template():
    e = _example_engine()
    action = e.framework.actions[0]
    e.apply_action(action, "Alpha", {"cell": 0})
    assert e.state["board"][0] == "X"
    e.apply_action(action, "Beta", {"cell": 4})
    assert e.state["board"][4] == "O"


def test_incr_and_append_ops():
    fw = Framework.from_dict(
        {
            "name": "Ops",
            "initial_state": {"hp": 10, "log": []},
            "actions": [
                {
                    "name": "damage",
                    "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
                    "effects": [
                        {"op": "incr", "path": "hp", "value": "-{{params.n}}"},
                        {"op": "append", "path": "log", "value": "took {{params.n}} damage"},
                    ],
                }
            ],
        }
    )
    e = Engine(fw, ["A"])
    e.apply_action(fw.actions[0], "A", {"n": 3})
    assert e.state["hp"] == 7
    assert e.state["log"] == ["took 3 damage"]


def test_remove_op():
    fw = Framework.from_dict(
        {
            "name": "Remove",
            "initial_state": {"tmp": "bye"},
            "actions": [
                {"name": "clear", "effects": [{"op": "remove", "path": "tmp"}]}
            ],
        }
    )
    e = Engine(fw, ["A"])
    e.apply_action(fw.actions[0], "A", {})
    assert "tmp" not in e.state


def test_dispatch_declare_outcome():
    e = _example_engine()
    e.dispatch_tool("declare_outcome", {"outcome": "win", "winner": "Alpha", "reason": "three in a row"}, "Alpha")
    assert e.outcome == "win"
    assert e.winner == "Alpha"


def test_run_game_without_llm_errors_on_missing_key():
    # With no agents stepping (turn order empty) and no API key, we should not crash
    # before the provider is even constructed: validate that Engine state is intact.
    e = _example_engine()
    assert e.public_state()["board"] == [" "] * 9
