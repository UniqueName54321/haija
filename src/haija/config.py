"""Project configuration: agents, model provider, and runtime settings.

A Haija project is a directory containing (at minimum) a ``haija.toml`` config
and a ``framework.json`` game definition.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass
class AgentSpec:
    """A playable agent: name, personality, and optional per-agent model."""

    name: str
    description: str = ""
    model: str | None = None

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
    framework_path: str = "framework.json"
    state_path: str = "state.json"
    max_steps_per_turn: int = 16
    path: Path | None = field(default=None, repr=False)

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
            )
            for a in data.get("agents", [])
        ]
        return cls(
            name=data.get("name", path.parent.name),
            agents=agents,
            model=model,
            framework_path=data.get("framework_path", "framework.json"),
            state_path=data.get("state_path", "state.json"),
            max_steps_per_turn=int(data.get("max_steps_per_turn", 16)),
            path=path,
        )


def default_config(name: str) -> str:
    """Return the TOML text for a freshly scaffolded project."""
    return f'''# Haija project configuration
name = "{name}"

# The agents that will play this game. Customize names, descriptions, and models.
[[agents]]
name = "Alpha"
description = "A careful, strategic player."

[[agents]]
name = "Beta"
description = "An aggressive, risk-taking player."

[model]
# Default model provider is OpenRouter (OpenAI-compatible API).
provider = "openrouter"
model = "{DEFAULT_MODEL}"
base_url = "{DEFAULT_BASE_URL}"
# api_key = "sk-or-..."                       # optional: hardcode (not recommended)
api_key_env = "{DEFAULT_API_KEY_ENV}"         # read the key from this env var
# model = "openai/gpt-4o-mini"                # override the model for every agent

# Optional: pin a specific model per agent by adding `model = "..."` to that agent.

# Runtime settings
max_steps_per_turn = 16
framework_path = "framework.json"
state_path = "state.json"
'''
