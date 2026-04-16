from __future__ import annotations

import uuid

from smollest.candidates import run_candidates
from smollest.compare import ComparisonResult, compare_outputs
from smollest.defaults import estimate_cost, infer_model_size_bucket
from smollest.metrics import compute_secondary_metrics
from smollest.results import log_result, print_comparison


def run_autocompare(
    *,
    project: str,
    provider: str,
    baseline_model: str,
    baseline_messages: list[dict],
    baseline_content: str,
    baseline_latency_ms: float,
    baseline_input_tokens: int,
    baseline_output_tokens: int,
    candidates: list[str],
    hf_token: str | None,
    trace_id: str | None = None,
    input_payload: dict | None = None,
) -> None:
    if not candidates:
        return
    baseline_cost = estimate_cost(
        baseline_model, baseline_input_tokens, baseline_output_tokens
    )
    candidate_results = run_candidates(
        messages=baseline_messages,
        candidates=candidates,
        hf_token=hf_token,
    )
    comparisons: list[ComparisonResult] = []
    candidate_latencies: dict[str, float] = {}
    active_trace_id = trace_id or str(uuid.uuid4())
    root_span_id = str(uuid.uuid4())
    baseline_secondary_metrics = compute_secondary_metrics(
        {
            "provider": provider,
            "model": baseline_model,
            "is_baseline": True,
            "content": baseline_content,
            "latency_ms": baseline_latency_ms,
            "input_tokens": baseline_input_tokens,
            "output_tokens": baseline_output_tokens,
            "cost": baseline_cost,
            "input_payload": input_payload or {},
            "trace_id": active_trace_id,
        }
    )
    for cr in candidate_results:
        candidate_latencies[cr.candidate] = cr.latency_ms
        if cr.error:
            comp = ComparisonResult(candidate=cr.candidate, error=cr.error)
        else:
            comp = compare_outputs(
                baseline=baseline_content,
                candidate_content=cr.content or "",
                candidate_name=cr.candidate,
            )
        comparisons.append(comp)
        candidate_cost = estimate_cost(cr.candidate, cr.input_tokens, cr.output_tokens)
        candidate_secondary_metrics = compute_secondary_metrics(
            {
                "provider": "candidate",
                "model": cr.candidate,
                "is_baseline": False,
                "content": cr.content,
                "latency_ms": cr.latency_ms,
                "input_tokens": cr.input_tokens,
                "output_tokens": cr.output_tokens,
                "cost": candidate_cost,
                "input_payload": input_payload or {},
                "trace_id": active_trace_id,
            }
        )
        log_result(
            project=project,
            provider=provider,
            baseline_model=baseline_model,
            baseline_model_size=infer_model_size_bucket(baseline_model),
            baseline_messages=baseline_messages,
            baseline_content=baseline_content,
            baseline_latency_ms=baseline_latency_ms,
            baseline_input_tokens=baseline_input_tokens,
            baseline_output_tokens=baseline_output_tokens,
            baseline_cost=baseline_cost,
            baseline_secondary_metrics=baseline_secondary_metrics,
            comparison=comp,
            candidate_content=cr.content,
            candidate_model_size=infer_model_size_bucket(cr.candidate),
            candidate_latency_ms=cr.latency_ms,
            candidate_input_tokens=cr.input_tokens,
            candidate_output_tokens=cr.output_tokens,
            candidate_cost=candidate_cost,
            candidate_secondary_metrics=candidate_secondary_metrics,
            trace_id=active_trace_id,
            parent_span_id=root_span_id,
            input_payload=input_payload or {},
        )
    print_comparison(
        baseline_model, baseline_latency_ms, comparisons, candidate_latencies
    )
