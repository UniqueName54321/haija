"""Model provider client. Default: OpenRouter (OpenAI-compatible API).

Stdlib-only (``urllib``), so Haija has zero hard dependencies. Any
OpenAI-compatible endpoint can be used by setting ``model.base_url``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""


class ProviderError(RuntimeError):
    """Raised on transport, auth, or response-shape errors."""


def reasoning_param(thinking: str | None) -> dict[str, Any] | None:
    """Map a per-agent thinking level to the provider's ``reasoning`` payload.

    None → no override (model default). "off" disables reasoning; "low"/
    "medium"/"high" set the reasoning effort (fast → deep). OpenRouter
    normalizes ``reasoning.effort`` across providers that support it.
    """
    if not thinking:
        return None
    level = str(thinking).strip().lower()
    if level == "off":
        return {"enabled": False}
    if level in ("low", "medium", "high"):
        return {"effort": level}
    return None


class ChatProvider:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = config.resolve_key()

    def _url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/UniqueName54321/haija",
            "X-Title": "Haija",
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        reasoning: dict[str, Any] | None = None,
    ) -> ChatResponse:
        if not self.api_key:
            raise ProviderError(
                f"No API key found. Set the {self.config.api_key_env} environment "
                "variable (or model.api_key in haija.toml)."
            )

        payload: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if reasoning:
            payload["reasoning"] = reasoning

        req = urllib.request.Request(
            self._url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise ProviderError(f"API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"Network error: {e.reason}") from e

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Unexpected response shape: {body}") from e

        content = message.get("content") or ""
        # OpenRouter exposes chain-of-thought via `reasoning`; some providers use
        # `reasoning_content`. Capture either so it can be exported as "thinking".
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        if isinstance(reasoning, list):
            reasoning = "".join(
                (p.get("text") or "") if isinstance(p, dict) else str(p)
                for p in reasoning
            )
        reasoning = reasoning or ""
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            args: dict[str, Any] = {}
            if isinstance(raw_args, str) and raw_args:
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )
        return ChatResponse(content=content, tool_calls=tool_calls, reasoning=reasoning)
