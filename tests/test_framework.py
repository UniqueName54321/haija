import json
import tomllib
from pathlib import Path

from haija.framework import Framework, assemble_framework, load_framework, normalize_content, validate_framework
from haija.generate import generate_framework
from haija.provider import ChatResponse
from haija.config import default_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_load_example_framework():
    fw = load_framework(EXAMPLES / "tic-tac-toe" / "framework.json")
    assert fw.name == "Tic-Tac-Toe"
    assert len(fw.actions) == 1
    assert fw.actions[0].name == "place_mark"
    assert fw.turn.order == "round_robin"
    assert fw.turn.max_turns == 9


def test_roundtrip_to_dict():
    fw = load_framework(EXAMPLES / "tic-tac-toe" / "framework.json")
    data = fw.to_dict()
    again = Framework.from_dict(data)
    assert again.name == fw.name
    assert [a.name for a in again.actions] == [a.name for a in fw.actions]


def test_generated_json_is_valid():
    # A framework generated with response_format=json_object should parse cleanly.
    text = '{"schema_version":"1","name":"Duel","initial_state":{"hp":10},"actions":[]}'
    fw = Framework.from_dict(json.loads(text))
    assert fw.name == "Duel"
    assert fw.initial_state["hp"] == 10


def test_normalize_content_coerces_types():
    c = normalize_content({
        "description": "d",
        "objective": None,
        "rules": "just one rule",
        "win_conditions": [1, 2],
        "lose_conditions": None,
        "initial_state": "not a dict",
        "actions": [
            {"name": "act", "effects": "bad"},
            {"name": "", "effects": []},
            {"description": "no name"},
            {"name": "ok", "parameters": None, "effects": [{"op": "set"}]},
        ],
        "turn": {"order": "weird", "max_turns": "15"},
    })
    assert c["rules"] == ["just one rule"]
    assert c["win_conditions"] == ["1", "2"]
    assert c["lose_conditions"] == []
    assert c["initial_state"] == {}
    assert [a["name"] for a in c["actions"]] == ["act", "ok"]
    assert c["actions"][0]["effects"] == []
    assert c["actions"][1]["parameters"] == {"type": "object", "properties": {}}
    assert c["turn"] == {"order": "round_robin", "max_turns": 15}


def test_normalize_content_handles_none():
    c = normalize_content(None)
    assert c["description"] == ""
    assert c["rules"] == []
    assert c["actions"] == []
    assert c["initial_state"] == {}
    assert c["turn"] == {"order": "round_robin", "max_turns": 20}


def test_assemble_framework_sets_envelope():
    fw = assemble_framework("My Project", {"objective": "win", "rules": ["r"]})
    assert fw.name == "My Project"
    assert fw.raw.get("schema_version") == "1"
    assert fw.objective == "win"
    assert fw.rules == ["r"]


def test_assemble_framework_name_fallback():
    fw = assemble_framework("", {"name": "Model's Name", "objective": "o"})
    assert fw.name == "Model's Name"
    fw2 = assemble_framework("", {})
    assert fw2.name == "Untitled Game"


def test_generate_framework_assembles_via_fake_provider():
    class Fake:
        def chat(self, messages, **kw):
            return ChatResponse(content=json.dumps({
                "description": "d",
                "objective": "o",
                "rules": ["r"],
                "initial_state": {"hp": 10},
                "actions": [{"name": "attack", "effects": []}],
            }))

    fw = generate_framework(Fake(), "make a game", name="proj")
    assert fw.name == "proj"
    assert fw.objective == "o"
    assert fw.rules == ["r"]
    assert fw.initial_state == {"hp": 10}
    assert [a.name for a in fw.actions] == ["attack"]


def test_semantic_framework_validation_rejects_bad_effect_graph():
    fw = Framework.from_dict({
        "name": "Bad", "initial_state": {"phase": "setup"},
        "actions": [{"name": "x", "effects": [{"op": "teleport", "path": "x"}]}],
    })
    errors, _ = validate_framework(fw)
    assert any("unknown effect op" in error for error in errors)
    assert any("setup phase requires" in error for error in errors)


def test_default_config_escapes_windows_paths_and_toml_special_characters():
    name = r'C:\Users\smurf\Games\"UNO"'
    assert tomllib.loads(default_config(name))["name"] == name
