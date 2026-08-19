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


def test_value_addressed_card_can_be_guarded_and_removed_from_hand():
    fw = {
        "name": "UNO",
        "initial_state": {"hands": {"Alpha": ["Wild", "Y0"]}, "discard_pile": []},
        "actions": [{
            "name": "play_card",
            "parameters": {"type": "object", "properties": {"card": {"type": "string"}}},
            "guard": [{"op": "exists", "path": "hands.{{actor}}.{{params.card}}"}],
            "effects": [
                {"op": "remove", "path": "hands.{{actor}}.{{params.card}}"},
                {"op": "append", "path": "discard_pile", "value": "{{params.card}}"},
            ],
        }],
    }
    e = Engine(Framework.from_dict(fw), ["Alpha"])
    result = e.apply_action(e.framework.actions[0], "Alpha", {"card": "Y0"})
    assert result["ok"] is True
    assert e.state["hands"]["Alpha"] == ["Wild"]
    assert e.state["discard_pile"] == ["Y0"]


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
    assert e.state["phase"] == "playing"


def test_engine_resolves_card_game_setup_before_turn_one():
    deck = [{"color": "red", "value": n} for n in range(30)]
    fw = {
        "name": "UNO",
        "description": "Each player starts with 7 cards.",
        "initial_state": {
            "phase": "setup",
            "players": [],
            "hands": {},
            "deck": deck,
            "discard_pile": [],
            "current_card": None,
            "current_color": None,
            "turn_index": 0,
        },
        "actions": [],
    }
    e = Engine(Framework.from_dict(fw), ["Alpha", "Beta", "Gamma"])
    assert e.state["phase"] == "playing"
    assert all(len(hand) == 7 for hand in e.state["hands"].values())
    assert len(e.state["deck"]) == 8
    assert len(e.state["discard_pile"]) == 1
    assert e.state["current_card"] == e.state["discard_pile"][-1]
    assert e.state["current_color"] == "red"


def test_uno_string_cards_initialize_discard_color_and_number():
    fw = {
        "name": "UNO",
        "description": "Each player starts with 2 cards.",
        "initial_state": {
            "phase": "setup",
            "hands": {},
            "deck": ["Wild", "RSkip", "Y0", "G8", "B2", "R4"],
            "discard_pile": [],
            "current_card": None,
            "current_color": "",
            "current_number": "",
        },
        "actions": [],
    }
    e = Engine(Framework.from_dict(fw), ["Alpha", "Beta"])
    top = e.state["discard_pile"][-1]
    assert top == e.state["current_card"]
    assert e.state["current_color"] == top[0]
    assert e.state["current_number"] == int(top[1:])


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


def test_generation_prompt_requires_playable_initial_state():
    from haija.generate import generate_framework

    class FakeProvider:
        def chat(self, messages, **kwargs):
            self.messages = messages
            from haija.provider import ChatResponse
            return ChatResponse(content='{"initial_state":{"phase":"playing"},"actions":[]}')

    provider = FakeProvider()
    generate_framework(provider, "make a card game", name="Cards")
    system = provider.messages[0]["content"]
    assert "immediately playable on turn 1" in system
    assert "Never emit an" in system
    assert "unresolved setup/loading/dealing phase" in system


def test_run_agent_stops_before_calling_provider():
    from haija.agents import run_agent
    from haija.config import AgentSpec, ProjectConfig

    class NeverCalledProvider:
        def chat(self, *args, **kwargs):
            raise AssertionError("provider should not be called after stop")

    e = _engine({"name": "G", "initial_state": {}, "actions": []})
    e.stop()
    cfg = ProjectConfig.load(
        Path(__file__).parent.parent / "examples" / "tic-tac-toe" / "haija.toml"
    )
    result = run_agent(NeverCalledProvider(), AgentSpec(name="Alpha"), e, cfg)
    assert result == {"content": "(stopped)"}


def test_provider_reports_clean_error_when_response_is_closed_by_stop(monkeypatch):
    from haija.config import ModelConfig
    from haija.provider import ChatProvider, ProviderError
    import pytest

    provider = ChatProvider(ModelConfig(api_key="test"))

    class InterruptedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            pass

        def read(self):
            provider.cancel()
            raise AttributeError("'NoneType' object has no attribute 'read'")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: InterruptedResponse())
    with pytest.raises(ProviderError, match="Stopped by user"):
        provider.chat([{"role": "user", "content": "go"}])
