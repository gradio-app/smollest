from __future__ import annotations

import threading
import time
from http.server import HTTPServer

from playwright.sync_api import sync_playwright


def test_dashboard_renders_and_add_model_prompt(tmp_path, monkeypatch):
    from smollest import results

    monkeypatch.setattr(results, "DATA_DIR", tmp_path)

    from smollest.compare import ComparisonResult
    from smollest.results import log_result
    from smollest.web import _make_handler

    log_result(
        project="ui",
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

    server = HTTPServer(("127.0.0.1", 0), _make_handler())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
            page.wait_for_timeout(200)
            assert "smollest" in page.title()
            assert page.locator("text=ui").count() >= 1

            page.on("dialog", lambda d: d.dismiss())
            page.locator("th.plus").click()
            time.sleep(0.2)
            browser.close()
    finally:
        server.shutdown()
