from __future__ import annotations

import os
import time

from mvlm.candidates import run_candidates
from mvlm.compare import ComparisonResult, compare_outputs
from mvlm.defaults import DEFAULT_CANDIDATES, estimate_cost
from mvlm.results import log_result, print_comparison


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

        active_candidates = (
            candidates if candidates is not None else self._wrapper._candidates
        )

        if not active_candidates:
            return response

        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        baseline_model = kwargs.get("model", "unknown")

        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(_anthropic_to_openai_messages(messages))

        baseline_content = ""
        for block in response.content:
            if block.type == "text":
                baseline_content += block.text

        usage = getattr(response, "usage", None)
        baseline_input_tokens = getattr(usage, "input_tokens", 0) or 0
        baseline_output_tokens = getattr(usage, "output_tokens", 0) or 0
        baseline_cost = estimate_cost(
            baseline_model, baseline_input_tokens, baseline_output_tokens
        )

        candidate_results = run_candidates(
            messages=openai_messages,
            candidates=active_candidates,
            hf_token=self._wrapper._hf_token,
        )

        comparisons = []
        candidate_latencies = {}
        for cr in candidate_results:
            candidate_latencies[cr.candidate] = cr.latency_ms
            if cr.error:
                comp = ComparisonResult(candidate=cr.candidate, error=cr.error)
            else:
                comp = compare_outputs(
                    baseline=baseline_content,
                    candidate_content=cr.content,
                    candidate_name=cr.candidate,
                )
            comparisons.append(comp)

            log_result(
                project=self._wrapper._project,
                baseline_model=baseline_model,
                baseline_content=baseline_content,
                baseline_latency_ms=baseline_latency,
                baseline_input_tokens=baseline_input_tokens,
                baseline_output_tokens=baseline_output_tokens,
                baseline_cost=baseline_cost,
                comparison=comp,
                candidate_latency_ms=cr.latency_ms,
                candidate_input_tokens=cr.input_tokens,
                candidate_output_tokens=cr.output_tokens,
                candidate_cost=estimate_cost(
                    cr.candidate, cr.input_tokens, cr.output_tokens
                ),
            )

        print_comparison(
            baseline_model, baseline_latency, comparisons, candidate_latencies
        )

        return response


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
        self._candidates = candidates if candidates is not None else DEFAULT_CANDIDATES
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._project = project
        self.messages = _Messages(self)

    def __getattr__(self, name):
        return getattr(self._client, name)
