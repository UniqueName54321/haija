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


def test_generation_repairs_validation_failure_and_rechecks():
    from haija.generate import generate_framework
    from haija.provider import ChatResponse

    invalid = {
        "initial_state": {},
        "actions": [{"name": "propose_change", "effects": [{"op": "move", "from": "deck"}]}],
    }
    repaired = {
        "initial_state": {"deck": [1], "proposals": []},
        "actions": [{"name": "propose_change", "effects": [
            {"op": "move", "from": "deck", "path": "proposals"}
        ]}],
    }

    class FakeProvider:
        def __init__(self):
            self.responses = [invalid, repaired]
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append(messages.copy())
            return ChatResponse(content=json.dumps(self.responses.pop(0)))

    provider = FakeProvider()
    fw = generate_framework(provider, "Nomic", name="Nomic")
    assert len(provider.calls) == 2
    repair_prompt = provider.calls[1][-1]["content"]
    assert "effect 'move' requires a path" in repair_prompt
    assert fw.actions[0].effects[0]["path"] == "proposals"


def test_generation_reports_repair_progress_events():
    from haija.generate import generate_framework

    outputs = iter([
        json.dumps({"initial_state": {}, "actions": [
            {"name": "bad", "effects": [{"op": "move", "from": "deck"}]}
        ]}),
        json.dumps({"initial_state": {}, "actions": []}),
    ])

    class FakeProvider:
        def chat_stream(self, messages, **kwargs):
            yield next(outputs)

    events = []
    generate_framework(FakeProvider(), "repair me", observer=events.append)
    types = [event["type"] for event in events]
    assert "generate_validation_failed" in types
    assert "generate_repair_start" in types
    assert types[-1] == "generate_done"


def test_provider_factory_and_provider_defaults(tmp_path):
    from haija.config import ModelConfig, ProjectConfig
    from haija.provider import AnthropicProvider, ChatProvider, CodexSubscriptionProvider, create_provider

    assert isinstance(create_provider(ModelConfig(provider="openai", api_key="x")), ChatProvider)
    assert isinstance(create_provider(ModelConfig(provider="anthropic", api_key="x")), AnthropicProvider)
    assert isinstance(create_provider(ModelConfig(provider="codex_subscription")), CodexSubscriptionProvider)

    config_path = tmp_path / "haija.toml"
    config_path.write_text('name = "T"\n[model]\nprovider = "anthropic"\n', encoding="utf-8")
    config = ProjectConfig.load(config_path)
    assert config.model.model == "claude-sonnet-5"
    assert config.model.base_url == "https://api.anthropic.com"
    assert config.model.api_key_env == "ANTHROPIC_API_KEY"


def test_openai_direct_uses_openai_headers_and_reasoning_effort():
    from haija.config import ModelConfig
    from haija.provider import ChatProvider

    provider = ChatProvider(ModelConfig(provider="openai", api_key="test"))
    headers = provider._headers()
    assert "HTTP-Referer" not in headers
    payload = {}
    provider._add_reasoning(payload, {"effort": "high"})
    assert payload == {"reasoning_effort": "high"}


def test_anthropic_adapter_converts_tools_and_tool_results():
    from haija.config import ModelConfig
    from haija.provider import AnthropicProvider

    provider = AnthropicProvider(ModelConfig(provider="anthropic", api_key="test"))
    payload = provider._payload(
        [
            {"role": "system", "content": "rules"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-1",
                "function": {"name": "play", "arguments": '{"card":"G4"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ],
        tools=[{"type": "function", "function": {
            "name": "play", "description": "Play", "parameters": {"type": "object"},
        }}],
    )
    assert payload["system"] == "rules"
    assert payload["messages"][0]["content"][0]["type"] == "tool_use"
    assert payload["messages"][1]["content"][0]["type"] == "tool_result"
    assert payload["tools"][0]["input_schema"] == {"type": "object"}
    assert payload["thinking"]["type"] == "adaptive"


def test_codex_subscription_provider_uses_ephemeral_read_only_cli(monkeypatch):
    from haija.config import ModelConfig
    from haija.provider import CodexSubscriptionProvider

    captured = {}

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            captured["command"] = command

        def communicate(self, prompt, timeout):
            captured["prompt"] = prompt
            output = Path(captured["command"][captured["command"].index("-o") + 1])
            output.write_text(
                json.dumps({
                    "content": "playing",
                    "reasoning": "",
                    "tool_calls": [{"id": "1", "name": "play", "arguments": {"x": 1}}],
                }),
                encoding="utf-8",
            )
            return "", ""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 1

    monkeypatch.setattr("haija.provider.shutil.which", lambda name: "codex")
    monkeypatch.setattr(
        "haija.provider.subprocess.run",
        lambda *args, **kwargs: type("Status", (), {
            "returncode": 0, "stdout": "Logged in using ChatGPT", "stderr": ""
        })(),
    )
    monkeypatch.setattr("haija.provider.subprocess.Popen", Process)
    provider = CodexSubscriptionProvider(
        ModelConfig(provider="codex_subscription", model="gpt-5.3-codex")
    )
    response = provider.chat([{"role": "user", "content": "go"}], tools=[])
    command = captured["command"]
    assert command[:5] == ["codex", "exec", "--ephemeral", "--sandbox", "read-only"]
    assert "--skip-git-repo-check" in command
    assert response.tool_calls[0].arguments == {"x": 1}


def test_codex_subscription_provider_rejects_api_key_login(monkeypatch):
    import pytest
    from haija.config import ModelConfig
    from haija.provider import CodexSubscriptionProvider, ProviderError

    monkeypatch.setattr("haija.provider.shutil.which", lambda name: "codex")
    monkeypatch.setattr(
        "haija.provider.subprocess.run",
        lambda *args, **kwargs: type("Status", (), {
            "returncode": 0, "stdout": "Logged in using an API key", "stderr": ""
        })(),
    )
    provider = CodexSubscriptionProvider(ModelConfig(provider="codex_subscription"))
    with pytest.raises(ProviderError, match="not signed in with ChatGPT"):
        provider.chat([{"role": "user", "content": "go"}])


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


def test_tool_capable_agent_stream_reconstructs_text_reasoning_and_calls(monkeypatch):
    from haija.config import ModelConfig
    from haija.provider import ChatProvider

    chunks = [
        {"choices": [{"delta": {"reasoning": "think "}}]},
        {"choices": [{"delta": {"content": "hello "}}]},
        {"choices": [{"delta": {"content": "world", "tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "send_", "arguments": "{\"to\":\"*\","}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "message", "arguments": "\"content\":\"hi\"}"}}]}}]},
    ]

    class Response:
        def __iter__(self):
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n".encode()
            yield b"data: [DONE]\n"

        def close(self):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    provider = ChatProvider(ModelConfig(api_key="test"))
    streamed = []
    response = provider.chat_streaming([{"role": "user", "content": "go"}], observer=streamed.append)
    assert response.content == "hello world"
    assert response.reasoning == "think "
    assert response.tool_calls[0].name == "send_message"
    assert response.tool_calls[0].arguments == {"to": "*", "content": "hi"}
    assert [event["kind"] for event in streamed] == ["reasoning", "content", "content"]


def _card_action():
    return {
        "name": "play_card",
        "parameters": {"type": "object", "properties": {"card": {"type": "string"}}, "required": ["card"]},
        "guard": [{"op": "exists", "path": "hands.{{actor}}.{{params.card}}"}],
        "effects": [
            {"op": "remove", "path": "hands.{{actor}}.{{params.card}}"},
            {"op": "append", "path": "discard_pile", "value": "{{params.card}}"},
        ],
    }


def test_state_scheduler_honors_reverse_skip_and_draw_cards():
    state = {
        "hands": {"A": ["BReverse", "R1"], "B": ["G1"], "C": ["Y1"]},
        "deck": ["R2", "R3", "R4", "R5"], "discard_pile": ["B5"],
        "turn_index": 0, "direction": 1,
    }
    e = Engine(Framework.from_dict({"name": "Cards", "initial_state": state, "actions": [_card_action()]}), ["A", "B", "C"])
    assert e.apply_action(e.framework.actions[0], "A", {"card": "BReverse"})["ok"]
    e.finish_turn()
    assert e.state["direction"] == -1 and e.current_actor() == "C"

    state["hands"]["A"] = ["RDraw2", "R1"]
    state["direction"] = 1
    e = Engine(Framework.from_dict({"name": "Cards", "initial_state": state, "actions": [_card_action()]}), ["A", "B", "C"])
    before = len(e.state["hands"]["B"])
    assert e.apply_action(e.framework.actions[0], "A", {"card": "RDraw2"})["ok"]
    e.finish_turn()
    assert len(e.state["hands"]["B"]) == before + 2 and e.current_actor() == "C"


def test_agent_state_view_hides_other_hands_and_deck():
    e = _engine({"name": "Cards", "initial_state": {
        "hands": {"Alpha": ["A"], "Beta": ["B", "C"]}, "deck": [1, 2, 3]}, "actions": []})
    view = e.public_state("Alpha")
    assert view["hands"]["Alpha"] == ["A"]
    assert view["hands"]["Beta"] == {"count": 2, "hidden": True}
    assert view["deck"] == {"count": 3, "hidden": True}


def test_agent_card_view_exposes_top_and_authoritative_turn_targets_without_history_orientation():
    e = Engine(Framework.from_dict({"name": "Cards", "initial_state": {
        "hands": {"A": ["R1"], "B": ["B1"], "C": ["G1"]},
        "discard_pile": ["GDraw2", "R5", "Y3"],
        "turn_index": 2, "direction": 1, "skip_count": 1,
    }, "actions": []}), ["A", "B", "C"])
    view = e.public_state("C")
    assert view["top_card"] == "GDraw2"
    assert view["discard_count"] == 3
    assert view["discard_pile"] == {"top": "GDraw2", "count": 3, "history_hidden": True}
    assert view["current_actor"] == "C"
    assert view["next_actor"] == "A"  # Draw/Skip target
    assert view["next_turn_actor"] == "B"  # actor after the pending skip


def test_empty_hand_wins_automatically():
    fw = Framework.from_dict({"name": "Cards", "initial_state": {
        "hands": {"A": ["R1"]}, "discard_pile": []}, "actions": [_card_action()]})
    e = Engine(fw, ["A"])
    assert e.apply_action(fw.actions[0], "A", {"card": "R1"})["ok"]
    assert (e.outcome, e.winner) == ("win", "A")


def test_action_effects_are_atomic_and_parameters_are_checked():
    fw = Framework.from_dict({"name": "Atomic", "initial_state": {"log": []}, "actions": [{
        "name": "act", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        "effects": [{"op": "append", "path": "log", "value": "changed"}, {"op": "append", "path": "missing", "value": "boom"}],
    }]})
    e = Engine(fw, ["A"])
    assert not e.apply_action(fw.actions[0], "A", {})["ok"]
    assert not e.apply_action(fw.actions[0], "A", {"n": "wrong"})["ok"]
    assert not e.apply_action(fw.actions[0], "A", {"n": 1})["ok"]
    assert e.state["log"] == []


def test_first_class_turn_and_draw_effects_resolve_in_order():
    fw = Framework.from_dict({"name": "Cards", "initial_state": {
        "hands": {"A": ["x"], "B": [], "C": []}, "deck": [1, 2, 3],
        "discard_pile": ["R1"], "turn_index": 0, "direction": 1,
    }, "actions": [{"name": "power", "effects": [
        {"op": "reverse_direction"},
        {"op": "draw", "from": "deck", "path": "hands.{{next_actor}}", "count": 2},
        {"op": "skip_next", "value": 1},
    ]}]})
    e = Engine(fw, ["A", "B", "C"])
    assert e.apply_action(fw.actions[0], "A", {})["ok"]
    assert len(e.state["hands"]["C"]) == 2
    e.finish_turn()
    assert e.current_actor() == "B"


def test_run_game_uses_authoritative_turn_index(monkeypatch):
    import haija.engine as engine_module
    from haija.config import AgentSpec, ModelConfig, ProjectConfig

    fw = Framework.from_dict({
        "name": "Turns", "initial_state": {"turn_index": 0, "direction": -1},
        "actions": [], "turn": {"order": "round_robin", "max_turns": 3},
    })
    e = Engine(fw, ["A", "B", "C"])
    seen = []
    monkeypatch.setattr(engine_module, "run_agent", lambda provider, agent, engine, cfg: seen.append(agent.name))
    monkeypatch.setattr(engine_module, "_persist", lambda engine, cfg: "saved")
    cfg = ProjectConfig("Turns", [AgentSpec(name=n) for n in e.agents], ModelConfig(api_key="x"))
    assert engine_module.run_game(e, object(), cfg, observer=lambda event: None) == "saved"
    assert seen == ["A", "C", "B"]


def test_repeated_identical_tool_calls_end_stalled_turn():
    from haija.agents import run_agent
    from haija.config import AgentSpec, ModelConfig, ProjectConfig
    from haija.provider import ChatResponse, ToolCall

    class Repeater:
        def chat(self, *args, **kwargs):
            return ChatResponse(content="again", tool_calls=[ToolCall("same", "get_state", {})])

    e = Engine(Framework.from_dict({"name": "G", "initial_state": {}, "actions": []}), ["A"])
    cfg = ProjectConfig("G", [AgentSpec(name="A")], ModelConfig(api_key="x"))
    result = run_agent(Repeater(), cfg.agents[0], e, cfg)
    assert result == {"content": "(stalled turn ended)"}


def test_uno_empty_generated_piles_are_built_and_dealt_without_setup_phase():
    fw = Framework.from_dict({
        "name": "UNO 3", "description": "Each player starts with 7 cards from a deck of 108 cards.",
        "initial_state": {"draw_pile": [], "discard_pile": [], "hands": {}, "players": {}, "current_color": "", "direction": 1},
        "actions": [],
    })
    e = Engine(fw, ["A", "B", "C", "D"])
    assert all(len(e.state["hands"][name]) == 7 for name in e.agents)
    assert len(e.state["draw_pile"]) == 79
    assert len(e.state["discard_pile"]) == 1
    assert e.state["current_color"] in ("Red", "Yellow", "Green", "Blue")


def test_generated_index_card_effects_use_pre_removal_card_and_inverted_draw_guard_is_repaired():
    reverse = {"color": "Blue", "number": None, "symbol": "Reverse"}
    fw = Framework.from_dict({
        "name": "UNO", "initial_state": {
            "hands": {"A": [reverse, {"color": "Red", "number": 1, "symbol": ""}], "B": []},
            "draw_pile": [{"color": "Green", "number": 2, "symbol": ""}],
            "discard_pile": [{"color": "Blue", "number": 5, "symbol": ""}],
            "current_color": "Blue", "direction": 1, "turn_index": 0,
        }, "actions": [{
            "name": "play_card", "parameters": {"type": "object", "properties": {"card_index": {"type": "integer"}}, "required": ["card_index"]},
            "guard": [{"op": "exists", "path": "hands.{{actor}}.{{params.card_index}}"}],
            "effects": [
                {"op": "append", "path": "discard_pile", "value": "hands.{{actor}}.{{params.card_index}}"},
                {"op": "remove", "path": "hands.{{actor}}.{{params.card_index}}"},
                {"op": "if", "guard": [{"op": "eq", "path": "hands.{{actor}}.{{params.card_index}}.symbol", "value": "Reverse"}], "effects": [{"op": "reverse_direction"}]},
            ],
        }, {
            "name": "draw_card", "effects": [{"op": "draw", "from": "draw_pile", "path": "hands.{{actor}}", "count": 1}],
            "guard": [{"op": "not_exists", "path": "draw_pile"}],
        }],
    })
    e = Engine(fw, ["A", "B"])
    assert e.apply_action(fw.actions[0], "A", {"card_index": 0})["ok"]
    assert e.state["discard_pile"][0] == reverse
    assert e.state["direction"] == -1
    assert e.apply_action(fw.actions[1], "A", {})["ok"]
    assert len(e.state["hands"]["A"]) == 2
