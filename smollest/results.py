from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smollest.compare import ComparisonResult

DATA_DIR = Path.home() / ".smollest"


def _get_project_file(project: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project)
    return DATA_DIR / f"{safe_name}.json"


def log_result(
    project: str,
    provider: str,
    baseline_model: str,
    baseline_model_size: str,
    baseline_messages: list[dict],
    baseline_content: str,
    baseline_latency_ms: float,
    baseline_input_tokens: int,
    baseline_output_tokens: int,
    baseline_cost: float | None,
    baseline_secondary_metrics: dict[str, float | int | str | bool | None],
    comparison: ComparisonResult,
    candidate_content: str | None,
    candidate_model_size: str,
    candidate_latency_ms: float,
    candidate_input_tokens: int,
    candidate_output_tokens: int,
    candidate_cost: float | None,
    candidate_secondary_metrics: dict[str, float | int | str | bool | None],
    trace_id: str,
    parent_span_id: str,
    input_payload: dict,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "provider": provider,
        "trace_id": trace_id,
        "parent_span_id": parent_span_id,
        "baseline_model": baseline_model,
        "baseline_model_size": baseline_model_size,
        "baseline_messages": baseline_messages,
        "baseline_content": baseline_content,
        "baseline_latency_ms": round(baseline_latency_ms, 1),
        "baseline_input_tokens": baseline_input_tokens,
        "baseline_output_tokens": baseline_output_tokens,
        "baseline_cost": baseline_cost,
        "baseline_secondary_metrics": baseline_secondary_metrics,
        "candidate": comparison.candidate,
        "candidate_content": candidate_content,
        "candidate_model_size": candidate_model_size,
        "candidate_latency_ms": round(candidate_latency_ms, 1),
        "candidate_input_tokens": candidate_input_tokens,
        "candidate_output_tokens": candidate_output_tokens,
        "candidate_cost": candidate_cost,
        "candidate_secondary_metrics": candidate_secondary_metrics,
        "score": comparison.score,
        "total_fields": comparison.total_fields,
        "matching_fields": comparison.matching_fields,
        "mismatched_fields": comparison.mismatched_fields,
        "error": comparison.error,
        "input_payload": input_payload,
    }

    log_file = _get_project_file(project)
    existing = []
    if log_file.exists():
        try:
            existing = json.loads(log_file.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(entry)
    log_file.write_text(json.dumps(existing, indent=2, default=str))


def print_comparison(
    baseline_model: str,
    baseline_latency_ms: float,
    comparisons: list[ComparisonResult],
    candidate_latencies: dict[str, float],
) -> None:
    print(f"\n{'=' * 60}")
    print(
        f"smollest comparison — baseline: {baseline_model} ({baseline_latency_ms:.0f}ms)"
    )
    print(f"{'=' * 60}")

    for comp in comparisons:
        latency = candidate_latencies.get(comp.candidate, 0)
        if comp.error:
            print(f"  {comp.candidate}: ERROR — {comp.error}")
        else:
            score_pct = f"{comp.score * 100:.0f}%" if comp.score is not None else "N/A"
            match_str = f"{len(comp.matching_fields)}/{comp.total_fields} fields"
            print(
                f"  {comp.candidate}: {score_pct} match ({match_str}) [{latency:.0f}ms]"
            )
            if comp.mismatched_fields:
                for m in comp.mismatched_fields:
                    print(
                        f"    ✗ {m['field']}: "
                        f"baseline={m['baseline']!r} vs candidate={m['candidate']!r}"
                    )

    print(f"{'=' * 60}\n")


def get_all_projects() -> list[str]:
    if not DATA_DIR.exists():
        return []
    projects = []
    for f in sorted(DATA_DIR.glob("*.json")):
        projects.append(f.stem)
    return projects


def get_project_data(project: str) -> list[dict]:
    log_file = _get_project_file(project)
    if not log_file.exists():
        return []
    try:
        return json.loads(log_file.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def report(project: str | None = None) -> None:
    if project:
        projects = [project]
    else:
        projects = get_all_projects()

    if not projects:
        print("No results found. Run some comparisons first.")
        return

    for proj in projects:
        entries = get_project_data(proj)
        if not entries:
            continue

        candidates: dict[str, list[float]] = {}
        for entry in entries:
            name = entry["candidate"]
            score = entry.get("score")
            if score is not None:
                candidates.setdefault(name, []).append(score)

        print(f"\n{'=' * 60}")
        print(f"smollest summary — project: {proj} ({len(entries)} comparisons)")
        print(f"{'=' * 60}")

        for name, scores in sorted(candidates.items()):
            avg = sum(scores) / len(scores)
            print(f"  {name}: {avg * 100:.1f}% avg match ({len(scores)} calls)")

        print(f"{'=' * 60}\n")
