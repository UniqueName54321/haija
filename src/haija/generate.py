"""Framework generation from a natural-language prompt."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from .framework import Framework, assemble_framework, framework_content_instructions, validate_framework
from .provider import ChatProvider

LOG = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("model output must be a JSON object")
        return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("model output must be a JSON object")
        return parsed
    raise ValueError("could not parse JSON from the model output")


def generate_framework(
    provider: ChatProvider,
    prompt: str,
    name: str = "",
    observer: Callable[[dict[str, Any]], None] | None = None,
    max_repair_attempts: int = 3,
) -> Framework:
    """Generate and validate a framework, repairing invalid model output."""
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

    max_repair_attempts = max(0, int(max_repair_attempts))
    LOG.info("generating framework (%s)", "streaming" if observer else "non-streaming")
    if observer:
        observer({"type": "generate_start", "prompt": prompt})

    last_errors: list[str] = []
    for attempt in range(max_repair_attempts + 1):
        if observer:
            full = ""
            for chunk in provider.chat_stream(messages):
                full += chunk
                observer({
                    "type": "generate_stream",
                    "chunk": chunk,
                    "full": full,
                    "attempt": attempt,
                })
        else:
            full = provider.chat(messages, json_mode=True).content

        try:
            data = _extract_json(full)
            fw = assemble_framework(name, data)
            errors, _ = validate_framework(fw)
            last_errors = list(errors)
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            fw = None
            last_errors = [str(exc)]

        if fw is not None and not last_errors:
            LOG.info(
                "framework generated: %d actions, %d rules (repairs=%d)",
                len(fw.actions), len(fw.rules), attempt,
            )
            if observer:
                observer({
                    "type": "generate_done",
                    "framework": fw.to_dict(),
                    "repair_attempts": attempt,
                })
            return fw

        LOG.warning(
            "framework validation failed (attempt %d/%d): %s",
            attempt + 1, max_repair_attempts + 1, "; ".join(last_errors),
        )
        if observer:
            observer({
                "type": "generate_validation_failed",
                "attempt": attempt,
                "errors": last_errors,
            })
        if attempt >= max_repair_attempts:
            break

        repair_number = attempt + 1
        if observer:
            observer({
                "type": "generate_repair_start",
                "attempt": repair_number,
                "max_attempts": max_repair_attempts,
            })
        messages.extend([
            {"role": "assistant", "content": full},
            {
                "role": "user",
                "content": (
                    "The framework failed executable validation.\n\n"
                    "Validation errors:\n- " + "\n- ".join(last_errors) + "\n\n"
                    "Repair only these validation errors while preserving the intended game "
                    "mechanics. Every effect must conform to the Haija effect schema. Return "
                    "the complete corrected framework JSON only."
                ),
            },
        ])

    raise ValueError(
        f"generated framework is not executable after {max_repair_attempts + 1} "
        f"attempt(s): {'; '.join(last_errors)}"
    )
