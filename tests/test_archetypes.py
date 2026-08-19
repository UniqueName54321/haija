import tempfile
from pathlib import Path

from haija.archetypes import BUILTIN_ARCHETYPES, Archetype
from haija.config import AgentSpec, ModelConfig, ProjectConfig, dump_config


def test_builtin_archetypes_present():
    ids = set(BUILTIN_ARCHETYPES)
    expected = {
        "normie", "chaos", "cheat", "baddie", "speedrunner",
        "completionist", "lawyer", "scientist", "contrarian",
    }
    assert expected <= ids


def test_resolve_persona_uses_archetype():
    cfg = ProjectConfig(name="g", agents=[], model=ModelConfig())
    agent = AgentSpec(name="A", archetype="chaos")
    assert "chaos" in cfg.resolve_persona(agent)


def test_resolve_persona_merges_description():
    cfg = ProjectConfig(name="g", agents=[], model=ModelConfig())
    agent = AgentSpec(name="A", archetype="normie", description="but secretly evil")
    p = cfg.resolve_persona(agent)
    assert "normally" in p
    assert "secretly evil" in p


def test_resolve_persona_custom_archetype():
    cfg = ProjectConfig(
        name="g", agents=[], model=ModelConfig(),
        archetypes={"mine": Archetype("mine", "Mine", "Custom persona")},
    )
    agent = AgentSpec(name="A", archetype="mine")
    assert cfg.resolve_persona(agent) == "Custom persona"


def test_enable_disable_archetype():
    cfg = ProjectConfig(name="g", agents=[], model=ModelConfig())
    cfg.enable_archetype("chaos")
    assert cfg.agent_for_archetype("chaos") is not None
    assert cfg.agent_for_archetype("chaos").name == "ChaosAgent"
    cfg.enable_archetype("chaos")  # idempotent
    assert len([a for a in cfg.agents if a.archetype == "chaos"]) == 1
    cfg.disable_archetype("chaos")
    assert cfg.agent_for_archetype("chaos") is None


def test_dump_roundtrip():
    cfg = ProjectConfig(
        name="My Game",
        agents=[
            AgentSpec(name="Alpha", archetype="normie"),
            AgentSpec(name="Beta", archetype="chaos", description="extra"),
        ],
        model=ModelConfig(),
        tone="gritty",
        archetypes={"mine": Archetype("mine", "Mine", "Does stuff.")},
    )
    text = dump_config(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "haija.toml"
        p.write_text(text, encoding="utf-8")
        loaded = ProjectConfig.load(p)
    assert loaded.name == "My Game"
    assert loaded.tone == "gritty"
    assert [a.name for a in loaded.agents] == ["Alpha", "Beta"]
    assert loaded.agents[0].archetype == "normie"
    assert loaded.agents[1].description == "extra"
    assert loaded.archetypes["mine"].persona == "Does stuff."
    assert loaded.model.model == ModelConfig().model


def test_load_parses_archetype_tone_custom():
    toml = '''name = "t"
tone = "silly"

[[agents]]
name = "X"
archetype = "lawyer"

[[archetypes]]
id = "my"
name = "My"
persona = "Persona"

[model]
provider = "openrouter"
model = "m"
base_url = "b"
api_key_env = "OPENROUTER_API_KEY"
'''
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "haija.toml"
        p.write_text(toml, encoding="utf-8")
        cfg = ProjectConfig.load(p)
    assert cfg.tone == "silly"
    assert cfg.agents[0].archetype == "lawyer"
    assert cfg.archetypes["my"].persona == "Persona"
    assert cfg.resolve_persona(cfg.agents[0]) == BUILTIN_ARCHETYPES["lawyer"].persona


def test_add_remove_custom_archetype():
    cfg = ProjectConfig(name="g", agents=[], model=ModelConfig())
    cfg.add_archetype("mine", "Mine", "Persona here")
    assert cfg.archetypes["mine"].persona == "Persona here"
    cfg.enable_archetype("mine")
    assert cfg.agent_for_archetype("mine") is not None
    cfg.remove_archetype("mine")
    assert "mine" not in cfg.archetypes
    assert cfg.agent_for_archetype("mine") is None


def test_set_tone_and_save():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "haija.toml"
        cfg = ProjectConfig(name="g", agents=[AgentSpec(name="A", archetype="normie")], model=ModelConfig())
        cfg.save(p)
        cfg2 = ProjectConfig.load(p)
        assert cfg2.name == "g"
        assert cfg2.agents[0].archetype == "normie"
        cfg2.set_tone("grim")
        cfg2.save(p)
        cfg3 = ProjectConfig.load(p)
        assert cfg3.tone == "grim"
