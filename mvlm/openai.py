from __future__ import annotations

import os
import time

from mvlm.candidates import run_candidates
from mvlm.compare import ComparisonResult, compare_outputs
from mvlm.defaults import DEFAULT_CANDIDATES, estimate_cost
from mvlm.results import log_result, print_comparison


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

        if not active_candidates:
            return response

        messages = kwargs.get("messages", [])
        baseline_content = response.choices[0].message.content
        baseline_model = kwargs.get("model", "unknown")

        usage = getattr(response, "usage", None)
        baseline_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        baseline_output_tokens = getattr(usage, "completion_tokens", 0) or 0
        baseline_cost = estimate_cost(
            baseline_model, baseline_input_tokens, baseline_output_tokens
        )

        candidate_results = run_candidates(
            messages=messages,
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
