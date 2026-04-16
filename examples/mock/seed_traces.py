from __future__ import annotations

import uuid

from smollest.compare import ComparisonResult
from smollest.results import log_result


def seed_trace(trace_input: str, baseline_output: str, project: str) -> None:
    trace_id = str(uuid.uuid4())
    base_messages = [{"role": "user", "content": trace_input}]
    candidates = [
        ("Qwen/Qwen3.5-3B-Instruct", baseline_output, 1.0),
        ("meta-llama/Llama-3.1-8B-Instruct", baseline_output.replace("2", "3"), 0.0),
    ]
    for model, output, score in candidates:
        comparison = ComparisonResult(
            candidate=model,
            score=score,
            total_fields=1,
            matching_fields=["answer"] if score == 1.0 else [],
            mismatched_fields=[]
            if score == 1.0
            else [{"field": "answer", "baseline": "2", "candidate": "3"}],
        )
        log_result(
            project=project,
            provider="openai",
            baseline_model="gpt-4.1",
            baseline_model_size="large",
            baseline_messages=base_messages,
            baseline_content=baseline_output,
            baseline_latency_ms=250.0,
            baseline_input_tokens=20,
            baseline_output_tokens=5,
            baseline_cost=0.00009,
            baseline_secondary_metrics={},
            comparison=comparison,
            candidate_content=output,
            candidate_model_size="small",
            candidate_latency_ms=110.0,
            candidate_input_tokens=20,
            candidate_output_tokens=5,
            candidate_cost=0.0,
            candidate_secondary_metrics={},
            trace_id=trace_id,
            parent_span_id=str(uuid.uuid4()),
            input_payload={"model": "gpt-4.1"},
        )


def main() -> None:
    project = "mock-traces"
    seed_trace('{"task":"math","question":"1+1?"}', '{"answer":"2"}', project)
    seed_trace('{"task":"math","question":"2+2?"}', '{"answer":"4"}', project)


if __name__ == "__main__":
    main()
