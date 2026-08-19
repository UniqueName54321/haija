import tempfile
from pathlib import Path

from haija.archetypes import BUILTIN_ARCHETYPES, Archetype
from haija.config import AgentSpec, ModelConfig, ProjectConfig, THINKING_LEVELS, dump_config
from haija.provider import reasoning_param
from haija.project import haija_home, projects_root


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


def test_agent_thinking_roundtrip():
    cfg = ProjectConfig(
        name="Think",
        agents=[
            AgentSpec(name="A", archetype="normie", thinking="high"),
            AgentSpec(name="B", archetype="chaos", thinking="off"),
            AgentSpec(name="C", archetype="scientist"),  # None
        ],
        model=ModelConfig(),
    )
    text = dump_config(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "haija.toml"
        p.write_text(text, encoding="utf-8")
        loaded = ProjectConfig.load(p)
    assert loaded.agents[0].thinking == "high"
    assert loaded.agents[1].thinking == "off"
    assert loaded.agents[2].thinking is None


def test_set_agent_thinking():
    cfg = ProjectConfig(name="g", agents=[AgentSpec(name="A"), AgentSpec(name="B")], model=ModelConfig())
    assert cfg.set_agent_thinking("A", "high")
    assert cfg.agents[0].thinking == "high"
    assert cfg.set_agent_thinking("A", None)
    assert cfg.agents[0].thinking is None
    assert not cfg.set_agent_thinking("Z", "low")
    cfg.set_agent_thinking("B", "default")
    assert cfg.agents[1].thinking is None
    cfg.set_agent_thinking("A", "bogus")
    assert cfg.agents[0].thinking is None  # invalid values → None


def test_reasoning_param():
    assert reasoning_param(None) is None
    assert reasoning_param("") is None
    assert reasoning_param("bogus") is None
    assert reasoning_param("off") == {"enabled": False}
    assert reasoning_param("low") == {"effort": "low"}
    assert reasoning_param("medium") == {"effort": "medium"}
    assert reasoning_param("high") == {"effort": "high"}


def test_thinking_levels_constant():
    assert THINKING_LEVELS == ("off", "low", "medium", "high")


def test_projects_root():
    root = projects_root()
    assert root.name == "projects"
    assert root.parent == haija_home()
