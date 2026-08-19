"""Model provider client. Default: OpenRouter (OpenAI-compatible API).

Stdlib-only (``urllib``), so Haija has zero hard dependencies. Any
OpenAI-compatible endpoint can be used by setting ``model.base_url``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig

LOG = logging.getLogger(__name__)


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
        self._cancelled = threading.Event()
        self._response: Any = None
        self._response_lock = threading.Lock()

    def cancel(self) -> None:
        """Cancel this provider and close any in-flight HTTP response."""
        self._cancelled.set()
        with self._response_lock:
            if self._response is not None:
                try:
                    self._response.close()
                except Exception:  # noqa: BLE001
                    pass

    def _check_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise ProviderError("Stopped by user.")

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
        self._check_cancelled()
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
                with self._response_lock:
                    self._response = resp
                raw = resp.read()
                self._check_cancelled()
                body = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            LOG.error("API error %d: %s", e.code, detail[:200])
            raise ProviderError(f"API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            LOG.error("Network error: %s", e.reason)
            raise ProviderError(f"Network error: {e.reason}") from e
        except Exception as e:  # closing a response from another thread varies by transport
            if self._cancelled.is_set():
                raise ProviderError("Stopped by user.") from e
            raise

        finally:
            with self._response_lock:
                self._response = None

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

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Stream chat completion over SSE. Yields content chunks."""
        self._check_cancelled()
        if not self.api_key:
            raise ProviderError(
                f"No API key found. Set the {self.config.api_key_env} "
                "environment variable (or model.api_key in haija.toml)."
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if reasoning:
            payload["reasoning"] = reasoning

        headers = self._headers()
        headers["Accept-Encoding"] = "identity"

        req = urllib.request.Request(
            self._url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=180)
            with self._response_lock:
                self._response = resp
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            LOG.error("API stream error %d: %s", e.code, detail[:200])
            raise ProviderError(f"API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            LOG.error("Network error (stream): %s", e.reason)
            raise ProviderError(f"Network error: {e.reason}") from e

        try:
            for line in resp:
                self._check_cancelled()
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
        except Exception as e:  # see cancellation handling in chat()
            if self._cancelled.is_set():
                raise ProviderError("Stopped by user.") from e
            raise
        finally:
            with self._response_lock:
                self._response = None
            resp.close()

    def chat_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
        observer=None,
    ) -> ChatResponse:
        """Stream a complete tool-capable response while reporting text deltas."""
        self._check_cancelled()
        if not self.api_key:
            raise ProviderError(
                f"No API key found. Set the {self.config.api_key_env} environment variable "
                "(or model.api_key in haija.toml)."
            )
        payload: dict[str, Any] = {
            "model": self.config.model, "messages": messages, "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if reasoning:
            payload["reasoning"] = reasoning
        headers = self._headers()
        headers["Accept-Encoding"] = "identity"
        req = urllib.request.Request(
            self._url(), data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            resp = urllib.request.urlopen(req, timeout=180)
            with self._response_lock:
                self._response = resp
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise ProviderError(f"API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"Network error: {e.reason}") from e

        content = ""
        thought = ""
        calls: dict[int, dict[str, str]] = {}
        try:
            for line in resp:
                self._check_cancelled()
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
                text_delta = delta.get("content") or ""
                if isinstance(text_delta, list):
                    text_delta = "".join(
                        (part.get("text") or "") if isinstance(part, dict) else str(part)
                        for part in text_delta
                    )
                if text_delta:
                    content += text_delta
                    if observer:
                        observer({"kind": "content", "delta": text_delta, "full": content})
                reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if isinstance(reasoning_delta, list):
                    reasoning_delta = "".join(
                        (part.get("text") or "") if isinstance(part, dict) else str(part)
                        for part in reasoning_delta
                    )
                if reasoning_delta:
                    thought += reasoning_delta
                    if observer:
                        observer({"kind": "reasoning", "delta": reasoning_delta, "full": thought})
                for tc in delta.get("tool_calls") or []:
                    index = int(tc.get("index", 0))
                    current = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        current["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    current["name"] += fn.get("name") or ""
                    current["arguments"] += fn.get("arguments") or ""
        except Exception as e:  # transport implementations differ when cancelled
            if self._cancelled.is_set():
                raise ProviderError("Stopped by user.") from e
            raise
        finally:
            with self._response_lock:
                self._response = None
            resp.close()

        tool_calls: list[ToolCall] = []
        for index in sorted(calls):
            raw = calls[index]
            try:
                arguments = json.loads(raw["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(raw["id"], raw["name"], arguments))
        return ChatResponse(content=content, tool_calls=tool_calls, reasoning=thought)
