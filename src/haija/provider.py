"""Model provider client. Default: OpenRouter (OpenAI-compatible API).

Stdlib-only (``urllib``), so Haija has zero hard dependencies. Any
OpenAI-compatible endpoint can be used by setting ``model.base_url``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.provider.strip().lower() == "openrouter":
            headers.update({
                "HTTP-Referer": "https://github.com/UniqueName54321/haija",
                "X-Title": "Haija",
            })
        return headers

    def _add_reasoning(
        self, payload: dict[str, Any], reasoning: dict[str, Any] | None
    ) -> None:
        if not reasoning:
            return
        if self.config.provider.strip().lower() == "openai":
            if reasoning.get("enabled") is not False and reasoning.get("effort"):
                payload["reasoning_effort"] = reasoning["effort"]
            return
        payload["reasoning"] = reasoning

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
        self._add_reasoning(payload, reasoning)

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
        self._add_reasoning(payload, reasoning)

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
        self._add_reasoning(payload, reasoning)
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


class AnthropicProvider(ChatProvider):
    """Anthropic Messages API adapter with Haija's provider interface."""

    def _url(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/v1/messages") else base + "/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": str(self.api_key or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    @staticmethod
    def _anthropic_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "system"
        )
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": str(message.get("content") or ""),
                    }],
                })
                continue
            content: Any = message.get("content") or ""
            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in message["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                content = blocks
            converted.append({"role": role, "content": content})
        return system, converted

    def _payload(self, messages, tools=None, reasoning=None, stream=False):
        system, converted = self._anthropic_input(messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": converted,
            "max_tokens": 16384,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
                for tool in tools
            ]
        model = self.config.model.lower()
        legacy_thinking = any(version in model for version in ("-4-5", "-4-0"))
        if reasoning and reasoning.get("enabled") is False:
            payload["thinking"] = {"type": "disabled"}
        elif legacy_thinking and reasoning:
            payload["thinking"] = {"type": "enabled", "budget_tokens": 2048}
        elif not legacy_thinking:
            payload["thinking"] = {"type": "adaptive", "display": "summarized"}
            if reasoning and reasoning.get("effort"):
                payload["output_config"] = {"effort": reasoning["effort"]}
        return payload

    def chat(self, messages, tools=None, json_mode=False, reasoning=None) -> ChatResponse:
        self._check_cancelled()
        if not self.api_key:
            raise ProviderError(f"No API key found. Set {self.config.api_key_env}.")
        req = urllib.request.Request(
            self._url(),
            json.dumps(self._payload(messages, tools, reasoning)).encode(),
            self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                with self._response_lock:
                    self._response = response
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            raise ProviderError(f"Anthropic API error {e.code}: {e.read().decode('utf-8', 'replace')}") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"Anthropic network error: {e.reason}") from e
        except Exception as e:
            if self._cancelled.is_set():
                raise ProviderError("Stopped by user.") from e
            raise
        finally:
            with self._response_lock:
                self._response = None
        text, thought, calls = "", "", []
        for block in body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "thinking":
                thought += block.get("thinking", "")
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(
                    block.get("id", ""), block.get("name", ""), block.get("input") or {}
                ))
        return ChatResponse(text, calls, thought)

    def chat_streaming(self, messages, tools=None, reasoning=None, observer=None) -> ChatResponse:
        self._check_cancelled()
        if not self.api_key:
            raise ProviderError(f"No API key found. Set {self.config.api_key_env}.")
        req = urllib.request.Request(
            self._url(),
            json.dumps(self._payload(messages, tools, reasoning, True)).encode(),
            self._headers(),
            method="POST",
        )
        try:
            response = urllib.request.urlopen(req, timeout=180)
            with self._response_lock:
                self._response = response
        except urllib.error.HTTPError as e:
            raise ProviderError(f"Anthropic API error {e.code}: {e.read().decode('utf-8', 'replace')}") from e
        except urllib.error.URLError as e:
            raise ProviderError(f"Anthropic network error: {e.reason}") from e
        text, thought, blocks = "", "", {}
        try:
            for line in response:
                self._check_cancelled()
                raw = line.decode().strip()
                if not raw.startswith("data: "):
                    continue
                try:
                    event = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue
                index = int(event.get("index", 0))
                if event.get("type") == "content_block_start":
                    block = event.get("content_block", {})
                    blocks[index] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "json": "",
                    }
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    piece = delta.get("text", "")
                    text += piece
                    if observer:
                        observer({"kind": "content", "delta": piece, "full": text})
                elif delta.get("type") == "thinking_delta":
                    piece = delta.get("thinking", "")
                    thought += piece
                    if observer:
                        observer({"kind": "reasoning", "delta": piece, "full": thought})
                elif delta.get("type") == "input_json_delta":
                    blocks.setdefault(index, {"id": "", "name": "", "json": ""})[
                        "json"
                    ] += delta.get("partial_json", "")
        except Exception as e:
            if self._cancelled.is_set():
                raise ProviderError("Stopped by user.") from e
            raise
        finally:
            with self._response_lock:
                self._response = None
            response.close()
        calls = []
        for block in blocks.values():
            if block["name"]:
                try:
                    args = json.loads(block["json"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(block["id"], block["name"], args))
        return ChatResponse(text, calls, thought)

    def chat_stream(self, messages, tools=None, reasoning=None):
        response = self.chat_streaming(messages, tools, reasoning)
        if response.content:
            yield response.content


class CodexSubscriptionProvider:
    """Use a locally signed-in Codex CLI session backed by a ChatGPT plan."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._cancelled = threading.Event()
        self._process: subprocess.Popen | None = None
        self._subscription_verified = False

    def cancel(self) -> None:
        self._cancelled.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def _verify_subscription_login(self) -> None:
        if self._subscription_verified:
            return
        try:
            status = subprocess.run(
                ["codex", "login", "status"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ProviderError(f"Could not verify Codex login: {e}") from e
        summary = f"{status.stdout}\n{status.stderr}".lower()
        if status.returncode != 0:
            detail = (status.stderr or status.stdout).strip()
            suffix = f" Details: {detail[-800:]}" if detail else ""
            raise ProviderError(
                "Could not verify the Codex login. Run `codex login` and try again."
                + suffix
            )
        if "chatgpt" not in summary:
            raise ProviderError(
                "Codex is not signed in with ChatGPT. Run `codex logout`, then "
                "`codex login` and choose ChatGPT subscription access."
            )
        self._subscription_verified = True

    def _run(self, messages, tools=None, observer=None) -> ChatResponse:
        if self._cancelled.is_set():
            raise ProviderError("Stopped by user.")
        if not shutil.which("codex"):
            raise ProviderError(
                "Codex CLI is not installed. Install it and run `codex login` first."
            )
        self._verify_subscription_login()
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {"type": "string"},
                "reasoning": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["id", "name", "arguments"],
                    },
                },
            },
            "required": ["content", "reasoning", "tool_calls"],
        }
        prompt = (
            "Act only as a Haija game-model backend. Do not inspect files or use your own tools. "
            "Given the conversation and available function schemas below, return the next assistant response. "
            "Select function calls exactly as an OpenAI-compatible model would.\n\nMESSAGES:\n"
            + json.dumps(messages)
            + "\n\nTOOLS:\n"
            + json.dumps(tools or [])
        )
        with tempfile.TemporaryDirectory(prefix="haija-codex-") as tmp:
            schema_path = str(Path(tmp) / "schema.json")
            output_path = str(Path(tmp) / "output.json")
            Path(schema_path).write_text(json.dumps(schema), encoding="utf-8")
            command = [
                "codex", "exec", "--ephemeral", "--sandbox", "read-only",
                "--skip-git-repo-check", "-C", tmp, "--output-schema", schema_path,
                "-o", output_path,
            ]
            if self.config.model:
                command += ["-m", self.config.model]
            command.append("-")
            process = None
            stderr = ""
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
                process = self._process
                _, stderr = self._process.communicate(prompt, timeout=300)
            except subprocess.TimeoutExpired as e:
                self.cancel()
                raise ProviderError("Codex CLI timed out.") from e
            except OSError as e:
                raise ProviderError(f"Could not start Codex CLI: {e}") from e
            finally:
                self._process = None
            if self._cancelled.is_set():
                raise ProviderError("Stopped by user.")
            if process is None or process.returncode != 0 or not Path(output_path).exists():
                raise ProviderError(
                    "Codex CLI failed. Run `codex login` and verify plan access. "
                    + stderr.strip()
                )
            try:
                data = json.loads(Path(output_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise ProviderError(f"Codex CLI returned invalid structured output: {e}") from e
        content, thought = data.get("content", ""), data.get("reasoning", "")
        if observer and thought:
            observer({"kind": "reasoning", "delta": thought, "full": thought})
        if observer and content:
            observer({"kind": "content", "delta": content, "full": content})
        calls = [
            ToolCall(call.get("id", ""), call.get("name", ""), call.get("arguments") or {})
            for call in data.get("tool_calls", [])
        ]
        return ChatResponse(content, calls, thought)

    def chat(self, messages, tools=None, json_mode=False, reasoning=None):
        return self._run(messages, tools)

    def chat_streaming(self, messages, tools=None, reasoning=None, observer=None):
        return self._run(messages, tools, observer)

    def chat_stream(self, messages, tools=None, reasoning=None):
        response = self._run(messages, tools)
        if response.content:
            yield response.content


def create_provider(config: ModelConfig):
    provider = config.provider.strip().lower()
    if provider == "anthropic":
        return AnthropicProvider(config)
    if provider in ("codex", "codex_subscription", "codex-subscription"):
        return CodexSubscriptionProvider(config)
    if provider in ("openrouter", "openai", "openai_compatible"):
        return ChatProvider(config)
    raise ProviderError(f"Unknown model provider '{config.provider}'.")
