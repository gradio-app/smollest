from __future__ import annotations

import os
import time
from collections.abc import Callable
from functools import wraps

from smollest.defaults import DEFAULT_CANDIDATES, get_default_candidates
from smollest.engine import run_autocompare

_PATCHED = False
_ORIGINAL_OPENAI_INIT = None
_AUTOCOMPARE_CONFIG = {
    "enabled": False,
    "project": "default",
    "candidates": None,
    "hf_token": None,
}


def _extract_content_from_chat_response(response) -> str:
    choice = response.choices[0] if getattr(response, "choices", None) else None
    if choice is None:
        return ""
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    return getattr(message, "content", "") or ""


def _extract_content_from_responses_response(response) -> str:
    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _wrap_method(
    method: Callable,
    *,
    parser: Callable[[dict], tuple[list[dict], str, dict]],
    response_parser: Callable[[object], str],
) -> Callable:
    @wraps(method)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        response = method(*args, **kwargs)
        if not _AUTOCOMPARE_CONFIG["enabled"]:
            return response
        baseline_latency = (time.monotonic() - start) * 1000
        messages, baseline_model, payload = parser(kwargs)
        usage = getattr(response, "usage", None)
        baseline_input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
            usage, "input_tokens", 0
        )
        baseline_output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
            usage, "output_tokens", 0
        )
        run_autocompare(
            project=_AUTOCOMPARE_CONFIG["project"],
            provider="openai",
            baseline_model=baseline_model,
            baseline_messages=messages,
            baseline_content=response_parser(response),
            baseline_latency_ms=baseline_latency,
            baseline_input_tokens=baseline_input_tokens,
            baseline_output_tokens=baseline_output_tokens,
            candidates=_AUTOCOMPARE_CONFIG["candidates"] or get_default_candidates(),
            hf_token=_AUTOCOMPARE_CONFIG["hf_token"] or os.environ.get("HF_TOKEN"),
            input_payload=payload,
        )
        return response

    return wrapper


def _parse_chat_payload(kwargs: dict) -> tuple[list[dict], str, dict]:
    return kwargs.get("messages", []), kwargs.get("model", "unknown"), kwargs


def _parse_responses_payload(kwargs: dict) -> tuple[list[dict], str, dict]:
    model = kwargs.get("model", "unknown")
    input_field = kwargs.get("input", "")
    if isinstance(input_field, str):
        messages = [{"role": "user", "content": input_field}]
    elif isinstance(input_field, list):
        messages = []
        for item in input_field:
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                text_bits = [
                    block.get("text", "")
                    for block in content
                    if block.get("type") in {"input_text", "text"}
                ]
                content = "\n".join(text_bits)
            messages.append({"role": role, "content": content})
    else:
        messages = [{"role": "user", "content": str(input_field)}]
    return messages, model, kwargs


def _instrument_openai_client(client) -> None:
    chat_create = getattr(client.chat.completions, "create", None)
    if callable(chat_create) and not getattr(
        chat_create, "__smollest_wrapped__", False
    ):
        wrapped = _wrap_method(
            chat_create,
            parser=_parse_chat_payload,
            response_parser=_extract_content_from_chat_response,
        )
        wrapped.__smollest_wrapped__ = True
        client.chat.completions.create = wrapped
    responses_api = getattr(client, "responses", None)
    responses_create = (
        getattr(responses_api, "create", None) if responses_api is not None else None
    )
    if callable(responses_create) and not getattr(
        responses_create, "__smollest_wrapped__", False
    ):
        wrapped = _wrap_method(
            responses_create,
            parser=_parse_responses_payload,
            response_parser=_extract_content_from_responses_response,
        )
        wrapped.__smollest_wrapped__ = True
        client.responses.create = wrapped


def autocompare(
    *,
    project: str = "default",
    candidates: list[str] | None = None,
    hf_token: str | None = None,
) -> None:
    global _PATCHED, _ORIGINAL_OPENAI_INIT
    _AUTOCOMPARE_CONFIG["enabled"] = True
    _AUTOCOMPARE_CONFIG["project"] = project
    _AUTOCOMPARE_CONFIG["candidates"] = candidates
    _AUTOCOMPARE_CONFIG["hf_token"] = hf_token
    if _PATCHED:
        return
    try:
        import openai as _openai
    except ImportError:
        raise ImportError(
            "openai package is required. Install it with: pip install smollest[openai]"
        )
    _ORIGINAL_OPENAI_INIT = _openai.OpenAI.__init__

    @wraps(_ORIGINAL_OPENAI_INIT)
    def patched_init(client_self, *args, **kwargs):
        _ORIGINAL_OPENAI_INIT(client_self, *args, **kwargs)
        _instrument_openai_client(client_self)

    _openai.OpenAI.__init__ = patched_init
    _PATCHED = True


autolog = autocompare


class _Completions:
    def __init__(self, wrapper: OpenAI):
        self._wrapper = wrapper

    def create(self, *, candidates: list[str] | None = None, **kwargs):
        start = time.monotonic()
        response = self._wrapper._client.chat.completions.create(**kwargs)
        baseline_latency = (time.monotonic() - start) * 1000
        active_candidates = (
            candidates if candidates is not None else self._wrapper._candidates
        )
        usage = getattr(response, "usage", None)
        run_autocompare(
            project=self._wrapper._project,
            provider="openai",
            baseline_model=kwargs.get("model", "unknown"),
            baseline_messages=kwargs.get("messages", []),
            baseline_content=_extract_content_from_chat_response(response),
            baseline_latency_ms=baseline_latency,
            baseline_input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            baseline_output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            candidates=active_candidates,
            hf_token=self._wrapper._hf_token,
            input_payload=kwargs,
        )
        return response


class _Chat:
    def __init__(self, wrapper: OpenAI):
        self.completions = _Completions(wrapper)


class OpenAI:
    def __init__(
        self,
        candidates: list[str] | None = None,
        hf_token: str | None = None,
        project: str = "default",
        **kwargs,
    ):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install smollest[openai]"
            )

        self._client = _openai.OpenAI(**kwargs)
        self._candidates = candidates if candidates is not None else DEFAULT_CANDIDATES
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._project = project
        self.chat = _Chat(self)

    def __getattr__(self, name):
        return getattr(self._client, name)
