"""Project configuration: agents, archetypes, tone, model provider, runtime settings.

A Haija project is a directory containing (at minimum) a ``haija.toml`` config
and a ``framework.json`` game definition. The config is read-only via ``load``
and can be written back via ``dump`` (used by the options menu).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .archetypes import BUILTIN_ARCHETYPES, Archetype

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"

# Per-agent thinking depth, mapped to the provider's `reasoning` parameter.
# "low" = fast/shallow, "medium" = balanced, "high" = deep/thorough, "off" = none.
THINKING_LEVELS = ("off", "low", "medium", "high")


@dataclass
class AgentSpec:
    """A playable agent: name, archetype, personality, and optional model."""

    name: str
    description: str = ""
    model: str | None = None
    archetype: str | None = None
    thinking: str | None = None  # None | "off" | "low" | "medium" | "high"

    def resolve_model(self, default: str) -> str:
        return self.model or default


@dataclass
class ModelConfig:
    """Model provider settings. Defaults to OpenRouter's OpenAI-compatible API."""

    provider: str = "openrouter"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    api_key_env: str = DEFAULT_API_KEY_ENV

    def resolve_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


@dataclass
class ProjectConfig:
    name: str
    agents: list[AgentSpec]
    model: ModelConfig
    tone: str = ""
    archetypes: dict[str, Archetype] = field(default_factory=dict)  # custom only
    framework_path: str = "framework.json"
    state_path: str = "state.json"
    max_steps_per_turn: int = 16
    path: Path | None = field(default=None, repr=False)

    # ---- archetypes -------------------------------------------------------
    def all_archetypes(self) -> dict[str, Archetype]:
        merged = dict(BUILTIN_ARCHETYPES)
        merged.update(self.archetypes)
        return merged

    def resolve_persona(self, agent: AgentSpec) -> str:
        parts: list[str] = []
        if agent.archetype:
            arch = self.all_archetypes().get(agent.archetype)
            if arch:
                parts.append(arch.persona)
        if agent.description:
            parts.append(agent.description)
        return " ".join(parts).strip() or "(no personality given)"

    def agent_for_archetype(self, arch_id: str) -> AgentSpec | None:
        return next((a for a in self.agents if a.archetype == arch_id), None)

    def enable_archetype(self, arch_id: str) -> None:
        if self.agent_for_archetype(arch_id):
            return
        arch = self.all_archetypes().get(arch_id)
        if not arch:
            return
        name = arch.name
        existing = {a.name for a in self.agents}
        base, i = name, 2
        while name in existing:
            name = f"{base} {i}"
            i += 1
        self.agents.append(AgentSpec(name=name, archetype=arch_id))

    def disable_archetype(self, arch_id: str) -> None:
        self.agents = [a for a in self.agents if a.archetype != arch_id]

    def set_archetype_enabled(self, arch_id: str, enabled: bool) -> None:
        if enabled:
            self.enable_archetype(arch_id)
        else:
            self.disable_archetype(arch_id)

    def set_tone(self, tone: str) -> None:
        self.tone = tone.strip()

    def set_agent_thinking(self, name: str, level: str | None) -> bool:
        """Set one agent's thinking depth; return False if the agent is unknown."""
        for a in self.agents:
            if a.name == name:
                a.thinking = _thinking(level) if level else None
                return True
        return False

    def add_archetype(self, arch_id: str, name: str, persona: str) -> None:
        self.archetypes[arch_id] = Archetype(arch_id, name or arch_id, persona)

    def remove_archetype(self, arch_id: str) -> None:
        self.archetypes.pop(arch_id, None)
        self.disable_archetype(arch_id)

    def save(self, path: Path | None = None) -> Path:
        p = path or self.path
        if p is None:
            raise ValueError("no path to save config to")
        p.write_text(dump_config(self), encoding="utf-8")
        return p

    # ---- load / dump ------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        model_raw = data.get("model", {})
        model = ModelConfig(
            provider=model_raw.get("provider", "openrouter"),
            model=model_raw.get("model", DEFAULT_MODEL),
            base_url=model_raw.get("base_url", DEFAULT_BASE_URL),
            api_key=model_raw.get("api_key"),
            api_key_env=model_raw.get("api_key_env", DEFAULT_API_KEY_ENV),
        )
        agents = [
            AgentSpec(
                name=a["name"],
                description=a.get("description", ""),
                model=a.get("model"),
                archetype=a.get("archetype"),
                thinking=_thinking(a.get("thinking")),
            )
            for a in data.get("agents", [])
        ]
        archetypes: dict[str, Archetype] = {}
        for a in data.get("archetypes", []):
            arch_id = a.get("id", "")
            if arch_id:
                archetypes[arch_id] = Archetype(
                    id=arch_id,
                    name=a.get("name", arch_id),
                    persona=a.get("persona", ""),
                )
        return cls(
            name=data.get("name", path.parent.name),
            agents=agents,
            model=model,
            tone=data.get("tone", ""),
            archetypes=archetypes,
            framework_path=data.get("framework_path", "framework.json"),
            state_path=data.get("state_path", "state.json"),
            max_steps_per_turn=int(data.get("max_steps_per_turn", 16)),
            path=path,
        )


def _thinking(value: Any) -> str | None:
    """Normalize a thinking level; unknown/invalid values fall back to None."""
    if value is None:
        return None
    v = str(value).strip().lower()
    return v if v in THINKING_LEVELS else None


def _tstr(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def dump_config(cfg: ProjectConfig) -> str:
    """Serialize a project config back to TOML (used by the options menu)."""
    L: list[str] = []
    L.append("# Haija project configuration")
    L.append(f"name = {_tstr(cfg.name)}")
    if cfg.tone:
        L.append(f"tone = {_tstr(cfg.tone)}")
    L.append(f"max_steps_per_turn = {cfg.max_steps_per_turn}")
    L.append(f"framework_path = {_tstr(cfg.framework_path)}")
    L.append(f"state_path = {_tstr(cfg.state_path)}")
    L.append("")

    for a in cfg.agents:
        L.append("[[agents]]")
        L.append(f"name = {_tstr(a.name)}")
        if a.archetype:
            L.append(f"archetype = {_tstr(a.archetype)}")
        if a.description:
            L.append(f"description = {_tstr(a.description)}")
        if a.model:
            L.append(f"model = {_tstr(a.model)}")
        if a.thinking:
            L.append(f"thinking = {_tstr(a.thinking)}")
        L.append("")

    for arch in cfg.archetypes.values():
        L.append("[[archetypes]]")
        L.append(f"id = {_tstr(arch.id)}")
        L.append(f"name = {_tstr(arch.name)}")
        L.append(f"persona = {_tstr(arch.persona)}")
        L.append("")

    m = cfg.model
    L.append("[model]")
    L.append(f"provider = {_tstr(m.provider)}")
    L.append(f"model = {_tstr(m.model)}")
    L.append(f"base_url = {_tstr(m.base_url)}")
    if m.api_key:
        L.append(f"api_key = {_tstr(m.api_key)}")
    L.append(f"api_key_env = {_tstr(m.api_key_env)}")
    L.append("")
    return "\n".join(L)


def default_config(name: str) -> str:
    """Return the TOML text for a freshly scaffolded project."""
    return f'''# Haija project configuration
name = "{name}"
# tone = "light-hearted"        # optional: a tone applied to ALL agents

max_steps_per_turn = 16
framework_path = "framework.json"
state_path = "state.json"

# The agents that will play this game. Each can use a built-in archetype
# (or a custom one from [[archetypes]] below), plus an optional description.
[[agents]]
name = "Alpha"
archetype = "normie"
# thinking = "high"               # per-agent thinking depth (off | low | medium | high)

[[agents]]
name = "Beta"
archetype = "chaos"
# thinking = "low"                # fast, shallow reasoning for this agent

# Built-in archetypes: normie, chaos, cheat, baddie, speedrunner,
# completionist, lawyer, scientist, contrarian.
# Define your own with [[archetypes]]:
# [[archetypes]]
# id = "my-agent"
# name = "MyAgent"
# persona = "Does exactly this one weird thing."

[model]
# Default model provider is OpenRouter (OpenAI-compatible API).
provider = "openrouter"
model = "{DEFAULT_MODEL}"
base_url = "{DEFAULT_BASE_URL}"
# api_key = "***"                       # optional: hardcode (not recommended)
api_key_env = "{DEFAULT_API_KEY_ENV}"         # read the key from this env var
# model = "openai/gpt-4o-mini"                # override the model for every agent
'''
