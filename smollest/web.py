from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

from smollest.candidates import run_candidates
from smollest.compare import ComparisonResult, compare_outputs
from smollest.defaults import estimate_cost, infer_model_size_bucket
from smollest.metrics import compute_secondary_metrics
from smollest.results import get_all_projects, get_project_data, log_result

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>smollest</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; display: grid; grid-template-columns: 240px 1fr; height: 100vh; background:#fffdf7; color:#222; }
    aside { border-right: 2px solid #222; padding: 18px 0; overflow:auto; }
    aside h1 { margin:0 18px 14px; font-size:20px; }
    aside a { display:block; padding:8px 18px; color:#333; text-decoration:none; border-left:3px solid transparent; }
    aside a.active { border-left-color:#222; font-weight:600; }
    main { padding: 20px; overflow:auto; }
    .tabs { display:flex; gap:8px; margin-bottom:12px; }
    .tab { border:1px solid #222; background:#fff; padding:6px 10px; cursor:pointer; }
    .tab.active { background:#222; color:#fff; }
    table { width:100%; border-collapse:collapse; border:2px solid #222; }
    th, td { border:1px solid #bbb; vertical-align:top; }
    th { background:#f7f1df; text-align:left; padding:8px; font-size:12px; text-transform:uppercase; }
    td { padding:0; }
    .cell-meta { font-size:12px; color:#555; padding:6px 8px; background:#faf6ea; border-bottom:1px solid #ddd; display:flex; gap:8px; flex-wrap:wrap; }
    .cell-body { padding:8px; white-space:pre-wrap; word-break:break-word; max-height:84px; overflow:hidden; cursor:pointer; font-family: ui-monospace, monospace; font-size:12px; }
    .cell-body.expanded { max-height:none; }
    .badge { border:1px solid #888; border-radius:2px; padding:0 4px; font-size:11px; }
    .size { font-weight:600; }
    .trace { border:1px solid #bbb; margin-bottom:12px; background:#fff; }
    .trace-head { padding:10px; background:#f7f1df; border-bottom:1px solid #bbb; display:flex; gap:12px; flex-wrap:wrap; font-size:12px; }
    .trace-body { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    .trace-col { border-right:1px solid #ddd; }
    .trace-col:last-child { border-right:none; }
    .trace-col h4 { margin:0; padding:8px; border-bottom:1px solid #ddd; font-size:12px; background:#faf6ea; }
    .trace-col pre { margin:0; padding:8px; font-size:12px; white-space:pre-wrap; word-break:break-word; max-height:220px; overflow:auto; }
    .muted { color:#666; font-size:12px; }
    .plus { cursor:pointer; font-size:20px; line-height:1; text-align:center; min-width:26px; }
  </style>
</head>
<body>
<aside>
  <h1>smollest</h1>
  <div id="projects"></div>
</aside>
<main id="main"></main>
<script>
let state = { data: __DATA_PLACEHOLDER__, currentProject: null, currentTab: "completions" };
const projectsEl = document.getElementById("projects");
const mainEl = document.getElementById("main");

function esc(v) { return (v || "").toString().replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function fmtCost(v) { if (v === null || v === undefined) return "—"; if (v < 0.001) return "<$0.001"; return "$" + v.toFixed(4); }
function fmtMetrics(metrics) { if (!metrics) return "—"; const keys = Object.keys(metrics); if (!keys.length) return "—"; return keys.map(k => `${k}: ${metrics[k]}`).join(" · "); }

function buildQueries(entries) {
  const grouped = {};
  for (const e of entries) {
    const key = e.trace_id || e.timestamp;
    if (!grouped[key]) grouped[key] = { trace_id: key, timestamp: e.timestamp, baseline: e, candidates: {} };
    grouped[key].candidates[e.candidate] = e;
  }
  return Object.values(grouped).sort((a,b) => b.timestamp.localeCompare(a.timestamp));
}

function renderProjects() {
  projectsEl.innerHTML = "";
  const names = Object.keys(state.data);
  for (const name of names) {
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = name;
    if (state.currentProject === name) a.classList.add("active");
    a.onclick = (ev) => { ev.preventDefault(); state.currentProject = name; renderProjects(); renderMain(); };
    projectsEl.appendChild(a);
  }
  if (!state.currentProject && names.length) state.currentProject = names[names.length - 1];
}

async function addModel(project, model) {
  await fetch("/api/add-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project, model }),
  });
  const res = await fetch("/api/data");
  state.data = await res.json();
  renderProjects();
  renderMain();
}

function renderCompletions(entries) {
  if (!entries.length) return "<div class='muted'>No data yet.</div>";
  const queries = buildQueries(entries);
  const baseline = queries[0].baseline.baseline_model;
  const models = [];
  for (const q of queries) {
    for (const key of Object.keys(q.candidates)) if (!models.includes(key)) models.push(key);
  }
  let head = "<tr><th>Time</th><th>" + esc(baseline) + "</th>";
  for (const m of models) {
    const any = entries.find(e => e.candidate === m);
    head += `<th>${esc(m)} <span class="size">(${esc(any?.candidate_model_size || "unknown")})</span></th>`;
  }
  head += `<th class="plus" title="Add model" onclick="const m=prompt('Model id (HF or OpenAI-compatible URL)'); if(m){addModel('${esc(state.currentProject)}', m);}">+</th></tr>`;
  let rows = "";
  for (const q of queries) {
    const b = q.baseline;
    rows += "<tr>";
    rows += `<td style="padding:8px; font-size:12px;">${new Date(q.timestamp).toLocaleString()}</td>`;
    rows += `<td><div class="cell-meta"><span>${(b.baseline_latency_ms||0).toFixed(0)}ms</span><span>${fmtCost(b.baseline_cost)}</span><span class="badge">${esc(b.baseline_model_size || "unknown")}</span><span>${fmtMetrics(b.baseline_secondary_metrics)}</span></div><div class="cell-body" onclick="this.classList.toggle('expanded')">${esc(b.baseline_content)}</div></td>`;
    for (const m of models) {
      const e = q.candidates[m];
      if (!e) {
        rows += "<td><div class='cell-body muted'>—</div></td>";
      } else {
        rows += `<td><div class="cell-meta"><span>${e.score === null || e.score === undefined ? "—" : (e.score*100).toFixed(0) + "% match"}</span><span>${(e.candidate_latency_ms||0).toFixed(0)}ms</span><span>${fmtCost(e.candidate_cost)}</span><span class="badge">${esc(e.candidate_model_size || "unknown")}</span><span>${fmtMetrics(e.candidate_secondary_metrics)}</span></div><div class="cell-body" onclick="this.classList.toggle('expanded')">${esc(e.error ? e.error : (e.candidate_content || "—"))}</div></td>`;
      }
    }
    rows += "<td></td></tr>";
  }
  return "<table><thead>" + head + "</thead><tbody>" + rows + "</tbody></table>";
}

function renderTraces(entries) {
  if (!entries.length) return "<div class='muted'>No traces yet.</div>";
  const queries = buildQueries(entries);
  let html = "";
  for (const q of queries) {
    const baseline = q.baseline;
    html += `<div class="trace"><div class="trace-head"><span>${new Date(q.timestamp).toLocaleString()}</span><span>${esc(baseline.provider || "unknown")}</span><span>${esc(baseline.baseline_model)}</span><span>${(baseline.baseline_latency_ms||0).toFixed(0)}ms</span><span>${fmtCost(baseline.baseline_cost)}</span></div><div class="trace-body">`;
    html += `<div class="trace-col"><h4>Input</h4><pre>${esc(JSON.stringify(baseline.baseline_messages || [], null, 2))}</pre></div>`;
    html += `<div class="trace-col"><h4>Baseline</h4><pre>${esc(baseline.baseline_content || "")}</pre></div>`;
    for (const e of Object.values(q.candidates)) {
      html += `<div class="trace-col"><h4>${esc(e.candidate)} (${esc(e.candidate_model_size || "unknown")})</h4><pre>${esc(e.error ? e.error : (e.candidate_content || ""))}</pre></div>`;
    }
    html += "</div></div>";
  }
  return html;
}

function renderMain() {
  const project = state.currentProject;
  if (!project) {
    mainEl.innerHTML = "<div class='muted'>No projects yet.</div>";
    return;
  }
  const entries = state.data[project] || [];
  const tabs = `<div class="tabs"><button class="tab ${state.currentTab === "completions" ? "active" : ""}" onclick="state.currentTab='completions';renderMain()">Completions</button><button class="tab ${state.currentTab === "traces" ? "active" : ""}" onclick="state.currentTab='traces';renderMain()">Traces</button></div>`;
  const body = state.currentTab === "traces" ? renderTraces(entries) : renderCompletions(entries);
  mainEl.innerHTML = `<h2>${esc(project)}</h2><div class="muted">${entries.length} comparison rows</div>${tabs}${body}`;
}

renderProjects();
renderMain();
</script>
</body>
</html>"""


def _load_data() -> dict[str, list[dict]]:
    data = {}
    for project in get_all_projects():
        data[project] = get_project_data(project)
    return data


def _replay_model_for_project(project: str, model: str) -> dict:
    entries = get_project_data(project)
    by_trace = {}
    for entry in entries:
        trace_id = entry.get("trace_id") or entry.get("timestamp")
        by_trace.setdefault(trace_id, entry)
    appended = 0
    for trace_id, baseline_entry in by_trace.items():
        messages = baseline_entry.get("baseline_messages") or []
        if not messages:
            continue
        baseline_content = baseline_entry.get("baseline_content", "")
        baseline_model = baseline_entry.get("baseline_model", "unknown")
        baseline_cost = baseline_entry.get("baseline_cost")
        baseline_latency = baseline_entry.get("baseline_latency_ms") or 0.0
        baseline_input_tokens = baseline_entry.get("baseline_input_tokens") or 0
        baseline_output_tokens = baseline_entry.get("baseline_output_tokens") or 0
        result = run_candidates(messages=messages, candidates=[model])
        if not result:
            continue
        candidate = result[0]
        if candidate.error:
            comparison = ComparisonResult(candidate=model, error=candidate.error)
        else:
            comparison = compare_outputs(
                baseline=baseline_content,
                candidate_content=candidate.content or "",
                candidate_name=model,
            )
        candidate_cost = estimate_cost(
            model, candidate.input_tokens, candidate.output_tokens
        )
        baseline_secondary_metrics = compute_secondary_metrics(
            {
                "provider": baseline_entry.get("provider", "unknown"),
                "model": baseline_model,
                "is_baseline": True,
                "content": baseline_content,
                "latency_ms": baseline_latency,
                "input_tokens": baseline_input_tokens,
                "output_tokens": baseline_output_tokens,
                "cost": baseline_cost,
                "trace_id": trace_id,
            }
        )
        candidate_secondary_metrics = compute_secondary_metrics(
            {
                "provider": "candidate",
                "model": model,
                "is_baseline": False,
                "content": candidate.content,
                "latency_ms": candidate.latency_ms,
                "input_tokens": candidate.input_tokens,
                "output_tokens": candidate.output_tokens,
                "cost": candidate_cost,
                "trace_id": trace_id,
            }
        )
        log_result(
            project=project,
            provider=baseline_entry.get("provider", "unknown"),
            baseline_model=baseline_model,
            baseline_model_size=baseline_entry.get("baseline_model_size", "unknown"),
            baseline_messages=messages,
            baseline_content=baseline_content,
            baseline_latency_ms=baseline_latency,
            baseline_input_tokens=baseline_input_tokens,
            baseline_output_tokens=baseline_output_tokens,
            baseline_cost=baseline_cost,
            baseline_secondary_metrics=baseline_secondary_metrics,
            comparison=comparison,
            candidate_content=candidate.content,
            candidate_model_size=infer_model_size_bucket(model),
            candidate_latency_ms=candidate.latency_ms,
            candidate_input_tokens=candidate.input_tokens,
            candidate_output_tokens=candidate.output_tokens,
            candidate_cost=candidate_cost,
            candidate_secondary_metrics=candidate_secondary_metrics,
            trace_id=trace_id,
            parent_span_id=baseline_entry.get("parent_span_id") or str(uuid.uuid4()),
            input_payload=baseline_entry.get("input_payload") or {},
        )
        appended += 1
    return {"added": appended}


def _make_handler():
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/data":
                body = json.dumps(_load_data(), default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            page = HTML_TEMPLATE.replace(
                "__DATA_PLACEHOLDER__", json.dumps(_load_data(), default=str)
            )
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/add-model":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            project = payload.get("project", "")
            model = payload.get("model", "")
            result = _replay_model_for_project(project=project, model=model)
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def show(port: int = 8765) -> None:
    server = HTTPServer(("127.0.0.1", port), _make_handler())
    url = f"http://127.0.0.1:{port}"
    print(f"smollest dashboard: {url}")
    print("Press Ctrl+C to stop")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
