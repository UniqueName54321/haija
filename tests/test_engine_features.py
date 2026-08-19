import json
from pathlib import Path

from haija.engine import Engine
from haija.framework import Framework
from haija.tools import all_tools


def _engine(fw_dict, agents=None):
    return Engine(Framework.from_dict(fw_dict), agents or ["Alpha", "Beta"])


def test_action_guard_rejects_illegal_move():
    fw = {
        "name": "G",
        "initial_state": {"board": [" ", " ", " "]},
        "actions": [{
            "name": "place",
            "parameters": {"type": "object", "properties": {"cell": {"type": "integer"}}},
            "guard": [{"op": "eq", "path": "board.{{params.cell}}", "value": " "}],
            "effects": [{"op": "set", "path": "board.{{params.cell}}", "value": "X"}],
        }],
    }
    e = _engine(fw)
    r = e.apply_action(e.framework.actions[0], "Alpha", {"cell": 0})
    assert r["ok"] is True
    assert e.state["board"][0] == "X"
    r2 = e.apply_action(e.framework.actions[0], "Alpha", {"cell": 0})
    assert r2["ok"] is False
    assert "illegal move" in r2["error"]
    assert e.state["board"][0] == "X"  # unchanged


def test_judge_declares_win():
    fw = {
        "name": "G",
        "initial_state": {"score": 0},
        "actions": [{"name": "score", "effects": [{"op": "incr", "path": "score", "value": 1}]}],
        "judge": {"win": [{"op": "gte", "path": "score", "value": 3}], "lose": [], "draw": []},
    }
    e = _engine(fw)
    e.apply_action(e.framework.actions[0], "Alpha", {})
    assert e.outcome is None
    e.apply_action(e.framework.actions[0], "Alpha", {})
    assert e.outcome is None
    e.apply_action(e.framework.actions[0], "Alpha", {})
    assert e.outcome == "win"
    assert e.winner == "Alpha"


def test_judge_removes_declare_outcome_tool():
    fw = {
        "name": "G",
        "initial_state": {},
        "actions": [],
        "judge": {"win": [{"op": "exists", "path": "winner"}]},
    }
    names = [t["function"]["name"] for t in all_tools(Framework.from_dict(fw))]
    assert "declare_outcome" not in names
    names2 = [t["function"]["name"] for t in all_tools(Framework.from_dict({"name": "G", "actions": []}))]
    assert "declare_outcome" in names2


def test_declare_outcome_rejected_when_judge_active():
    fw = {
        "name": "G",
        "initial_state": {},
        "actions": [],
        "judge": {"win": [{"op": "exists", "path": "winner"}]},
    }
    e = _engine(fw)
    r = e.dispatch_tool("declare_outcome", {"outcome": "win", "winner": "Alpha", "reason": "x"}, "Alpha")
    assert "determines the outcome automatically" in r
    assert e.outcome is None


def test_private_memory():
    e = _engine({"name": "G", "initial_state": {}, "actions": []})
    e.dispatch_tool("remember", {"key": "secret", "value": "xyz"}, "Alpha")
    e.dispatch_tool("remember", {"key": "plan", "value": 42}, "Beta")
    assert e.private["Alpha"]["secret"] == "xyz"
    assert e.private["Beta"]["plan"] == 42
    assert json.loads(e.dispatch_tool("recall", {"key": "secret"}, "Alpha")) == "xyz"
    assert json.loads(e.dispatch_tool("recall", {"key": "secret"}, "Beta")) is None
    assert json.loads(e.dispatch_tool("recall_all", {}, "Beta")) == {"plan": 42}


def test_alliances():
    e = _engine({"name": "G", "initial_state": {}, "actions": []}, agents=["A", "B", "C"])
    e.dispatch_tool("form_alliance", {"with_agent": "B"}, "A")
    assert ("A", "B") in e.alliances
    assert json.loads(e.dispatch_tool("my_allies", {}, "A")) == ["B"]
    assert json.loads(e.dispatch_tool("my_allies", {}, "B")) == ["A"]
    assert json.loads(e.dispatch_tool("list_alliances", {}, "A")) == [["A", "B"]]
    e.dispatch_tool("break_alliance", {"with_agent": "B"}, "A")
    assert e.alliances == set()
    assert json.loads(e.dispatch_tool("my_allies", {}, "A")) == []


def test_engine_injects_players_into_initial_state():
    fw = {
        "name": "G",
        "initial_state": {"phase": "setup", "players": [], "hands": {}},
        "actions": [],
    }
    e = Engine(Framework.from_dict(fw), ["Alpha", "Beta", "Gamma"])
    assert e.state["players"] == ["Alpha", "Beta", "Gamma"]
    assert e.state["hands"] == {"Alpha": [], "Beta": [], "Gamma": []}


def test_engine_stop_flag():
    e = _engine({"name": "G", "initial_state": {"x": 0}, "actions": []})
    assert not e.stopped
    e.stop()
    assert e.stopped


def test_engine_does_not_overwrite_preset_players():
    fw = {
        "name": "G",
        "initial_state": {"players": ["PreSet"]},
        "actions": [],
    }
    e = Engine(Framework.from_dict(fw), ["Alpha", "Beta"])
    assert e.state["players"] == ["PreSet"]  # not overwritten


def test_general_config_roundtrip():
    from haija.config import ProjectConfig, dump_config
    from io import StringIO
    import tomllib

    cfg = ProjectConfig.load(
        Path(__file__).parent.parent / "examples" / "tic-tac-toe" / "haija.toml"
    )
    cfg.general.log_level = "debug"
    cfg.general.log_file = "/tmp/test.log"
    toml_text = dump_config(cfg)
    data = tomllib.loads(toml_text)
    assert data.get("general", {}).get("log_level") == "debug"
    assert data.get("general", {}).get("log_file") == "/tmp/test.log"


def test_generate_framework_observer():
    from haija.framework import assemble_framework
    from haija.generate import generate_framework

    # We can't call the real API, but we can test that observer is called
    # by passing a mock-like provider that raises an error after streaming
    class FakeProvider:
        def chat_stream(self, messages, tools=None, reasoning=None):
            yield "{"
            yield '"description"'
            yield ': "test"'
            yield "}"

    events = []
    try:
        generate_framework(FakeProvider(), "test", name="T", observer=events.append)
    except Exception:
        pass
    types = [e["type"] for e in events]
    assert "generate_start" in types
    assert "generate_stream" in types
    assert "generate_done" in types
    # The last event should have a framework
    done_ev = [e for e in events if e["type"] == "generate_done"]
    assert len(done_ev) == 1
    assert "framework" in done_ev[0]
