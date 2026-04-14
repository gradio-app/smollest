from __future__ import annotations

import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

from mvlm.results import get_all_projects, get_project_data

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mvlm — Minimum Viable Language Model</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; height: 100vh; background: #f5f5f5; color: #333; }

  .sidebar { width: 240px; background: #1a1a2e; color: #eee; padding: 20px 0; overflow-y: auto; flex-shrink: 0; }
  .sidebar h1 { font-size: 20px; padding: 0 20px 16px; border-bottom: 1px solid #333; font-weight: 600; }
  .sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #888; padding: 16px 20px 8px; }
  .sidebar a { display: block; padding: 8px 20px; color: #ccc; text-decoration: none; font-size: 14px; transition: background 0.15s; }
  .sidebar a:hover { background: #16213e; }
  .sidebar a.active { background: #0f3460; color: #fff; font-weight: 500; }

  .main { flex: 1; overflow-y: auto; padding: 32px; }
  .main h2 { font-size: 24px; margin-bottom: 8px; }
  .main .subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }

  .metrics { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .metric-card { background: #fff; border-radius: 8px; padding: 16px 20px; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .metric-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .metric-card .detail { font-size: 12px; color: #666; margin-top: 2px; }
  .green { color: #16a34a; }
  .blue { color: #2563eb; }
  .orange { color: #ea580c; }

  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th { background: #f8f9fa; text-align: left; padding: 12px 16px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #666; border-bottom: 2px solid #e5e7eb; }
  td { padding: 10px 16px; border-bottom: 1px solid #f0f0f0; font-size: 13px; vertical-align: top; }
  tr:hover td { background: #f8fafc; }
  .truncate { max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
  .truncate.expanded { white-space: pre-wrap; word-break: break-word; }
  .score { font-weight: 600; }
  .score-100 { color: #16a34a; }
  .score-high { color: #65a30d; }
  .score-mid { color: #ea580c; }
  .score-low { color: #dc2626; }
  .error-badge { background: #fef2f2; color: #dc2626; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
  .cost { color: #666; font-size: 12px; }
  .latency { color: #888; font-size: 12px; }
  .empty { text-align: center; padding: 60px; color: #999; }
</style>
</head>
<body>
<div class="sidebar">
  <h1>mvlm</h1>
  <h2>Projects</h2>
  <div id="project-list"></div>
</div>
<div class="main" id="main">
  <div class="empty">Select a project to view results</div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const projects = Object.keys(DATA);
const sidebar = document.getElementById('project-list');
const main = document.getElementById('main');

function formatCost(c) {
  if (c === null || c === undefined) return '—';
  if (c < 0.001) return '<$0.001';
  return '$' + c.toFixed(4);
}

function scoreClass(s) {
  if (s === null || s === undefined) return '';
  if (s >= 1.0) return 'score-100';
  if (s >= 0.7) return 'score-high';
  if (s >= 0.4) return 'score-mid';
  return 'score-low';
}

function renderProject(name) {
  document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
  document.querySelector(`.sidebar a[data-project="${name}"]`)?.classList.add('active');

  const entries = DATA[name] || [];
  if (!entries.length) {
    main.innerHTML = '<div class="empty">No results yet for this project.</div>';
    return;
  }

  const candidates = {};
  let totalCalls = 0;
  entries.forEach(e => {
    if (!candidates[e.candidate]) candidates[e.candidate] = { scores: [], latencies: [], costs: [] };
    if (e.score !== null) candidates[e.candidate].scores.push(e.score);
    candidates[e.candidate].latencies.push(e.candidate_latency_ms);
    if (e.candidate_cost !== null) candidates[e.candidate].costs.push(e.candidate_cost);
    totalCalls++;
  });

  const baselineModel = entries[0]?.baseline_model || '?';
  const avgBaselineLatency = entries.reduce((s, e) => s + (e.baseline_latency_ms || 0), 0) / entries.length;
  const totalBaselineCost = entries.reduce((s, e) => s + (e.baseline_cost || 0), 0);

  let metricsHtml = `
    <div class="metric-card">
      <div class="label">Baseline</div>
      <div class="value blue">${baselineModel}</div>
      <div class="detail">${avgBaselineLatency.toFixed(0)}ms avg · ${formatCost(totalBaselineCost)} total</div>
    </div>
    <div class="metric-card">
      <div class="label">Total Calls</div>
      <div class="value">${totalCalls}</div>
    </div>`;

  for (const [cand, data] of Object.entries(candidates)) {
    const avg = data.scores.length ? (data.scores.reduce((a,b)=>a+b,0) / data.scores.length * 100).toFixed(0) : '?';
    const avgLat = (data.latencies.reduce((a,b)=>a+b,0) / data.latencies.length).toFixed(0);
    const totalCost = data.costs.reduce((a,b)=>a+b,0);
    metricsHtml += `
      <div class="metric-card">
        <div class="label">${cand}</div>
        <div class="value ${Number(avg) >= 80 ? 'green' : Number(avg) >= 50 ? 'orange' : ''}">${avg}%</div>
        <div class="detail">${avgLat}ms avg · ${formatCost(totalCost)} total</div>
      </div>`;
  }

  let rowsHtml = '';
  entries.forEach((e, i) => {
    const sc = e.score !== null ? (e.score * 100).toFixed(0) + '%' : '—';
    const mismatch = (e.mismatched_fields || []).map(m => m.field).join(', ') || '—';
    rowsHtml += `<tr>
      <td>${new Date(e.timestamp).toLocaleString()}</td>
      <td><span class="truncate" onclick="this.classList.toggle('expanded')">${(e.baseline_content || '').replace(/</g,'&lt;')}</span></td>
      <td>${e.candidate}</td>
      <td><span class="score ${scoreClass(e.score)}">${sc}</span></td>
      <td class="latency">${(e.baseline_latency_ms||0).toFixed(0)}ms</td>
      <td class="latency">${(e.candidate_latency_ms||0).toFixed(0)}ms</td>
      <td class="cost">${formatCost(e.baseline_cost)}</td>
      <td class="cost">${formatCost(e.candidate_cost)}</td>
      <td>${e.error ? '<span class="error-badge">'+e.error+'</span>' : mismatch}</td>
    </tr>`;
  });

  main.innerHTML = `
    <h2>${name}</h2>
    <div class="subtitle">${entries.length} comparisons · baseline: ${baselineModel}</div>
    <div class="metrics">${metricsHtml}</div>
    <table>
      <thead><tr>
        <th>Time</th><th>Baseline Output</th><th>Candidate</th><th>Match</th>
        <th>Baseline Latency</th><th>Candidate Latency</th>
        <th>Baseline Cost</th><th>Candidate Cost</th><th>Mismatches / Errors</th>
      </tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>`;
}

projects.forEach(name => {
  const a = document.createElement('a');
  a.href = '#';
  a.textContent = name;
  a.dataset.project = name;
  a.onclick = (ev) => { ev.preventDefault(); renderProject(name); };
  sidebar.appendChild(a);
});

if (projects.length) renderProject(projects[projects.length - 1]);
</script>
</body>
</html>"""


def _make_handler(html: str):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, format, *args):
            pass

    return Handler


def show(port: int = 8765) -> None:
    projects = get_all_projects()
    data = {}
    for p in projects:
        data[p] = get_project_data(p)

    page = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(data, default=str))

    server = HTTPServer(("127.0.0.1", port), _make_handler(page))

    url = f"http://127.0.0.1:{port}"
    print(f"mvlm dashboard: {url}")
    print("Press Ctrl+C to stop")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
