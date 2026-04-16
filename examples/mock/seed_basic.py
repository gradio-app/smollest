from __future__ import annotations

import uuid

from smollest.compare import ComparisonResult
from smollest.results import log_result


def main() -> None:
    project = "mock-basic"
    baseline_messages = [
        {"role": "system", "content": "Return compact JSON."},
        {"role": "user", "content": "Classify: I loved the service."},
    ]
    for i in range(3):
        trace_id = str(uuid.uuid4())
        baseline_content = '{"label":"positive","confidence":0.9}'
        for candidate_name, score in [
            ("Qwen/Qwen3.5-3B-Instruct", 1.0),
            ("meta-llama/Llama-3.1-8B-Instruct", 0.5),
        ]:
            comparison = ComparisonResult(
                candidate=candidate_name,
                score=score,
                total_fields=2,
                matching_fields=["label"] if score < 1.0 else ["label", "confidence"],
                mismatched_fields=[]
                if score == 1.0
                else [{"field": "confidence", "baseline": 0.9, "candidate": 0.7}],
            )
            log_result(
                project=project,
                provider="openai",
                baseline_model="gpt-4.1-mini",
                baseline_model_size="small",
                baseline_messages=baseline_messages,
                baseline_content=baseline_content,
                baseline_latency_ms=180 + i * 11,
                baseline_input_tokens=34,
                baseline_output_tokens=12,
                baseline_cost=0.00004,
                baseline_secondary_metrics={},
                comparison=comparison,
                candidate_content='{"label":"positive","confidence":0.7}',
                candidate_model_size="small",
                candidate_latency_ms=120 + i * 8,
                candidate_input_tokens=34,
                candidate_output_tokens=12,
                candidate_cost=0.0,
                candidate_secondary_metrics={},
                trace_id=trace_id,
                parent_span_id=str(uuid.uuid4()),
                input_payload={"model": "gpt-4.1-mini"},
            )


if __name__ == "__main__":
    main()
