import json
from pathlib import Path

from haija.framework import Framework, load_framework

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
