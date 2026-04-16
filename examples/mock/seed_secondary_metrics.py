from __future__ import annotations

import uuid

from smollest.compare import ComparisonResult
from smollest.results import log_result


def main() -> None:
    project = "mock-secondary-metrics"
    messages = [{"role": "user", "content": "Extract topic from this paragraph."}]
    trace_id = str(uuid.uuid4())
    baseline = "topic=mlops"
    models = [
        ("mistralai/Mistral-Small-24B-Instruct-2501", "large", 240.0, 0.00011, 0.024),
        ("Qwen/Qwen3.5-3B-Instruct", "small", 90.0, 0.0, 0.007),
    ]
    for model, size, latency_ms, cost, co2 in models:
        log_result(
            project=project,
            provider="anthropic",
            baseline_model="claude-sonnet-4-20250514",
            baseline_model_size="large",
            baseline_messages=messages,
            baseline_content=baseline,
            baseline_latency_ms=330.0,
            baseline_input_tokens=98,
            baseline_output_tokens=15,
            baseline_cost=0.0005,
            baseline_secondary_metrics={"co2_g": 0.041},
            comparison=ComparisonResult(candidate=model, score=1.0, total_fields=1),
            candidate_content=baseline,
            candidate_model_size=size,
            candidate_latency_ms=latency_ms,
            candidate_input_tokens=98,
            candidate_output_tokens=15,
            candidate_cost=cost,
            candidate_secondary_metrics={"co2_g": co2},
            trace_id=trace_id,
            parent_span_id=str(uuid.uuid4()),
            input_payload={"model": "claude-sonnet-4-20250514"},
        )


if __name__ == "__main__":
    main()
