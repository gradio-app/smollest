from __future__ import annotations

import os
import time
from functools import wraps

from smollest.defaults import get_default_candidates
from smollest.engine import run_autocompare

_PATCHED = False
_ORIGINAL_ANTHROPIC_INIT = None
_AUTOCOMPARE_CONFIG = {
    "enabled": False,
    "project": "default",
    "candidates": None,
    "hf_token": None,
}


def _anthropic_to_openai_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        converted.append({"role": msg["role"], "content": content})
    return converted


class _Messages:
    def __init__(self, wrapper: Anthropic):
        self._wrapper = wrapper

    def create(self, *, candidates: list[str] | None = None, **kwargs):
        start = time.monotonic()
        response = self._wrapper._client.messages.create(**kwargs)
        baseline_latency = (time.monotonic() - start) * 1000
        openai_messages = []
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(_anthropic_to_openai_messages(messages))

        baseline_content = ""
        for block in response.content:
            if block.type == "text":
                baseline_content += block.text

        usage = getattr(response, "usage", None)
        run_autocompare(
            project=self._wrapper._project,
            provider="anthropic",
            baseline_model=kwargs.get("model", "unknown"),
            baseline_messages=openai_messages,
            baseline_content=baseline_content,
            baseline_latency_ms=baseline_latency,
            baseline_input_tokens=getattr(usage, "input_tokens", 0) or 0,
            baseline_output_tokens=getattr(usage, "output_tokens", 0) or 0,
            candidates=candidates
            if candidates is not None
            else self._wrapper._candidates,
            hf_token=self._wrapper._hf_token,
            input_payload=kwargs,
        )
        return response


def autocompare(
    *,
    project: str = "default",
    candidates: list[str] | None = None,
    hf_token: str | None = None,
) -> None:
    global _PATCHED, _ORIGINAL_ANTHROPIC_INIT
    _AUTOCOMPARE_CONFIG["enabled"] = True
    _AUTOCOMPARE_CONFIG["project"] = project
    _AUTOCOMPARE_CONFIG["candidates"] = candidates
    _AUTOCOMPARE_CONFIG["hf_token"] = hf_token
    if _PATCHED:
        return
    try:
        import anthropic as _anthropic
    except ImportError:
        raise ImportError(
            "anthropic package is required. Install it with: pip install smollest[anthropic]"
        )
    _ORIGINAL_ANTHROPIC_INIT = _anthropic.Anthropic.__init__

    @wraps(_ORIGINAL_ANTHROPIC_INIT)
    def patched_init(client_self, *args, **kwargs):
        _ORIGINAL_ANTHROPIC_INIT(client_self, *args, **kwargs)
        _instrument_anthropic_client(client_self)

    _anthropic.Anthropic.__init__ = patched_init
    _PATCHED = True


autolog = autocompare


def _instrument_anthropic_client(client) -> None:
    create = getattr(client.messages, "create", None)
    if not callable(create) or getattr(create, "__smollest_wrapped__", False):
        return

    @wraps(create)
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        response = create(*args, **kwargs)
        if not _AUTOCOMPARE_CONFIG["enabled"]:
            return response
        baseline_latency = (time.monotonic() - start) * 1000
        messages = kwargs.get("messages", [])
        openai_messages = []
        system = kwargs.get("system")
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(_anthropic_to_openai_messages(messages))
        baseline_content = ""
        for block in response.content:
            if block.type == "text":
                baseline_content += block.text
        usage = getattr(response, "usage", None)
        run_autocompare(
            project=_AUTOCOMPARE_CONFIG["project"],
            provider="anthropic",
            baseline_model=kwargs.get("model", "unknown"),
            baseline_messages=openai_messages,
            baseline_content=baseline_content,
            baseline_latency_ms=baseline_latency,
            baseline_input_tokens=getattr(usage, "input_tokens", 0) or 0,
            baseline_output_tokens=getattr(usage, "output_tokens", 0) or 0,
            candidates=_AUTOCOMPARE_CONFIG["candidates"] or get_default_candidates(),
            hf_token=_AUTOCOMPARE_CONFIG["hf_token"] or os.environ.get("HF_TOKEN"),
            input_payload=kwargs,
        )
        return response

    wrapped.__smollest_wrapped__ = True
    client.messages.create = wrapped


class Anthropic:
    def __init__(
        self,
        candidates: list[str] | None = None,
        hf_token: str | None = None,
        project: str = "default",
        **kwargs,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install it with: pip install smollest[anthropic]"
            )

        self._client = _anthropic.Anthropic(**kwargs)
        self._candidates = (
            candidates if candidates is not None else get_default_candidates()
        )
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._project = project
        self.messages = _Messages(self)

    def __getattr__(self, name):
        return getattr(self._client, name)
