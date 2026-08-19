"""Framework generation from a natural-language prompt."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from .framework import Framework, assemble_framework, framework_content_instructions
from .provider import ChatProvider

LOG = logging.getLogger(__name__)


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


def generate_framework(
    provider: ChatProvider,
    prompt: str,
    name: str = "",
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> Framework:
    """Generate a framework from a prompt. Pass an *observer* to stream."""
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

    if observer is None:
        LOG.info("generating framework (non-streaming)")
        resp = provider.chat(messages, json_mode=True)
        data = _extract_json(resp.content)
        fw = assemble_framework(name, data)
        LOG.info("framework generated: %d actions, %d rules", len(fw.actions), len(fw.rules))
        return fw

    LOG.info("generating framework (streaming)")
    observer({"type": "generate_start", "prompt": prompt})
    full = ""
    for chunk in provider.chat_stream(messages):
        full += chunk
        observer({"type": "generate_stream", "chunk": chunk, "full": full})

    data = _extract_json(full)
    fw = assemble_framework(name, data)
    LOG.info("framework generated: %d actions, %d rules", len(fw.actions), len(fw.rules))
    observer({"type": "generate_done", "framework": fw.to_dict()})
    return fw
