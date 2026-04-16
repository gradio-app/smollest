from __future__ import annotations

import json
from unittest.mock import patch


def test_add_model_replays_and_appends(tmp_path, monkeypatch):
    from smollest import results

    monkeypatch.setattr(results, "DATA_DIR", tmp_path)

    from smollest.compare import ComparisonResult
    from smollest.results import log_result

    log_result(
        project="p",
        provider="openai",
        baseline_model="gpt-4.1-mini",
        baseline_model_size="small",
        baseline_messages=[{"role": "user", "content": "hi"}],
        baseline_content='{"a": 1}',
        baseline_latency_ms=10.0,
        baseline_input_tokens=1,
        baseline_output_tokens=1,
        baseline_cost=0.0,
        baseline_secondary_metrics={},
        comparison=ComparisonResult(candidate="c1", score=1.0, total_fields=1),
        candidate_content='{"a": 1}',
        candidate_model_size="small",
        candidate_latency_ms=10.0,
        candidate_input_tokens=1,
        candidate_output_tokens=1,
        candidate_cost=0.0,
        candidate_secondary_metrics={},
        trace_id="t1",
        parent_span_id="s1",
        input_payload={},
    )

    from smollest.web import _replay_model_for_project

    class _CR:
        def __init__(self):
            self.candidate = "new"
            self.content = '{"a": 1}'
            self.error = None
            self.latency_ms = 1.0
            self.input_tokens = 1
            self.output_tokens = 1

    with patch("smollest.web.run_candidates", return_value=[_CR()]):
        out = _replay_model_for_project(project="p", model="new")
        assert out["added"] == 1

    data = json.loads((tmp_path / "p.json").read_text())
    assert any(e["candidate"] == "new" for e in data)
