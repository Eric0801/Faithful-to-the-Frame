#!/usr/bin/env python3
"""Shared OpenAI-compatible chat-completion helpers for harness scripts."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_COMPAT_SEED = 2_147_483_647


class OpenAICompatibleTransientError(RuntimeError):
    """Retryable API failure for an OpenAI-compatible chat-completions endpoint."""


class OpenAICompatiblePermanentError(RuntimeError):
    """Non-retryable API failure for an OpenAI-compatible chat-completions endpoint."""


class UrllibResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.text = body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class UrllibSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    def post(self, url: str, *, data: str, timeout: float) -> UrllibResponse:
        request = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return UrllibResponse(response.status, response.read())
        except urllib.error.HTTPError as exc:
            return UrllibResponse(exc.code, exc.read())
        except TimeoutError as exc:
            raise UrllibRequestsShim.Timeout(str(exc)) from exc
        except socket.timeout as exc:
            raise UrllibRequestsShim.Timeout(str(exc)) from exc
        except urllib.error.URLError as exc:
            raise UrllibRequestsShim.ConnectionError(str(exc.reason)) from exc


class UrllibRequestsShim:
    class RequestException(RuntimeError):
        pass

    class Timeout(RequestException):
        pass

    class ConnectionError(RequestException):
        pass

    @staticmethod
    def Session() -> UrllibSession:
        return UrllibSession()


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    response_id: str | None
    response_model: str | None
    usage: dict[str, Any] | None
    response_json: dict[str, Any]


def load_requests() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        return UrllibRequestsShim
    return requests


def resolve_api_key(explicit_api_key: str, api_key_env_var: str) -> str:
    candidate = explicit_api_key.strip()
    if candidate:
        return candidate
    from_env = os.environ.get(api_key_env_var, "").strip()
    if from_env:
        return from_env
    raise OpenAICompatiblePermanentError(
        f"missing API key: pass --api-key or set {api_key_env_var}"
    )


def normalize_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    normalized = abs(int(seed)) % MAX_COMPAT_SEED
    if normalized == 0:
        normalized = 1
    return normalized


def extract_text_from_chat_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatiblePermanentError("chat response missing non-empty choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise OpenAICompatiblePermanentError("chat response choice is not an object")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise OpenAICompatiblePermanentError("chat response did not contain text content")


def response_error_message(response: Any) -> str:
    body = response.text.strip()
    try:
        parsed = response.json()
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                body = message.strip()
    if not body:
        body = f"HTTP {response.status_code}"
    body = " ".join(body.split())
    if len(body) > 400:
        body = body[:397] + "..."
    return body


class OpenAICompatibleClient:
    """Minimal chat-completions client for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.requests = load_requests()
        self.session = self.requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat_completion(
        self,
        *,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        normalized_seed = normalize_seed(seed)
        if normalized_seed is not None:
            payload["seed"] = normalized_seed
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"{self.api_base_url}/chat/completions"
        try:
            response = self.session.post(
                url,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
        except self.requests.Timeout as exc:
            raise OpenAICompatibleTransientError(f"request timed out: {exc}") from exc
        except self.requests.ConnectionError as exc:
            raise OpenAICompatibleTransientError(f"connection failed: {exc}") from exc
        except self.requests.RequestException as exc:
            raise OpenAICompatiblePermanentError(f"request failed: {exc}") from exc

        if response.status_code >= 400:
            message = response_error_message(response)
            error = f"HTTP {response.status_code}: {message}"
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise OpenAICompatibleTransientError(error)
            raise OpenAICompatiblePermanentError(error)

        try:
            payload_json = response.json()
        except ValueError as exc:
            raise OpenAICompatiblePermanentError("response body is not valid JSON") from exc
        if not isinstance(payload_json, dict):
            raise OpenAICompatiblePermanentError("response body is not a JSON object")

        text = extract_text_from_chat_response(payload_json)
        return ChatCompletionResult(
            text=text,
            response_id=payload_json.get("id") if isinstance(payload_json.get("id"), str) else None,
            response_model=payload_json.get("model")
            if isinstance(payload_json.get("model"), str)
            else None,
            usage=payload_json.get("usage") if isinstance(payload_json.get("usage"), dict) else None,
            response_json=payload_json,
        )
