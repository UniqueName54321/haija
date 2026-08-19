"""Framework generation from a natural-language prompt."""

from __future__ import annotations

import json
import re

from .framework import Framework, assemble_framework, framework_content_instructions
from .provider import ChatProvider


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("could not parse JSON from the model output")


def generate_framework(provider: ChatProvider, prompt: str, name: str = "") -> Framework:
    system = (
        "You are Haija's game designer. Given a prompt, produce the game-design "
        "content for a Haija framework. Haija fills in the name, schema version, "
        "and structural normalization itself.\n\n"
        + framework_content_instructions()
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resp = provider.chat(messages, json_mode=True)
    data = _extract_json(resp.content)
    return assemble_framework(name, data)
