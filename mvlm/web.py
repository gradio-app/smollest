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
<title>smollest — Minimum Viable Language Model</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  svg.filters { position: absolute; width: 0; height: 0; }

  body {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    display: flex; height: 100vh;
    background: #fffef8;
    color: #333;
  }

  .sidebar {
    width: 220px;
    background: #fffef8;
    padding: 20px 0;
    overflow-y: auto;
    flex-shrink: 0;
    border-right: 2.5px solid #333;
  }
  .sidebar h1 { font-size: 22px; padding: 0 18px 14px; border-bottom: 2px solid #333; letter-spacing: -0.5px; }
  .sidebar h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; color: #888; padding: 16px 18px 6px; }
  .sidebar a { display: block; padding: 7px 18px; color: #555; text-decoration: none; font-size: 15px; border-left: 3px solid transparent; transition: all 0.1s; }
  .sidebar a:hover { color: #000; border-left-color: #999; }
  .sidebar a.active { color: #000; border-left-color: #333; }

  .main { flex: 1; overflow-y: auto; padding: 32px 40px; }
  .main h2 { font-size: 26px; margin-bottom: 4px; }
  .main .subtitle { color: #777; margin-bottom: 24px; font-size: 14px; }

  .top-row { display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; align-items: flex-start; }
  .metric-card {
    background: #fffef8;
    border: 2.5px solid #333;
    border-radius: 3px;
    padding: 14px 18px;
    min-width: 140px;
  }
  .metric-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { font-size: 28px; margin-top: 2px; }
  .metric-card .detail { font-size: 13px; color: #777; margin-top: 2px; }

  .chart-card {
    background: #fffef8;
    border: 2.5px solid #333;
    border-radius: 3px;
    padding: 14px 18px;
    min-width: 220px;
    flex: 1;
  }
  .chart-card .chart-title { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .bar-row { display: flex; align-items: center; margin-bottom: 6px; }
  .bar-row:last-child { margin-bottom: 0; }
  .bar-label { width: 110px; font-size: 12px; color: #555; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0; }
  .bar-track { flex: 1; height: 18px; background: #f0ece0; border: 1.5px solid #999; border-radius: 2px; position: relative; margin: 0 8px; }
  .bar-fill { height: 100%; border-radius: 1px; }
  .bar-fill.green { background: #2a9d3e; color: #2a9d3e; }
  .bar-fill.blue { background: #3a7dd8; color: #3a7dd8; }
  .bar-fill.orange { background: #d4740a; color: #d4740a; }
  .bar-fill.red { background: #c92a2a; color: #c92a2a; }
  .bar-fill.baseline-fill { background: #555; }
  .bar-val { font-size: 12px; color: #555; min-width: 70px; text-align: right; flex-shrink: 0; white-space: nowrap; }

  .green { color: #2a9d3e; }
  .blue { color: #3a7dd8; }
  .orange { color: #d4740a; }

  table {
    width: 100%;
    border-collapse: collapse;
    background: #fffef8;
    border: 2.5px solid #333;
    font-family: Arial, Helvetica, sans-serif;
  }
  th {
    background: #faf8f0;
    text-align: left;
    padding: 10px 14px;
    height: 72px;
    font-size: 13px;
    line-height: 1.15;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #555;
    border-bottom: 2.5px solid #333;
    border-right: 1.5px solid #ccc;
    vertical-align: middle;
  }
  th:last-child { border-right: none; }
  td {
    padding: 0;
    border-bottom: 2px solid #bbb;
    border-right: 1.5px solid #ccc;
    font-size: 14px;
    vertical-align: top;
  }
  td:last-child { border-right: none; }
  .cell-header {
    background: #faf8f0;
    padding: 6px 12px;
    font-size: 12px;
    color: #555;
    border-bottom: 1px solid #ddd;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
  }
  .cell-header .score-badge {
    padding: 0 6px;
    border-radius: 2px;
    font-size: 12px;
    line-height: 1.2;
  }
  .cell-header .score-badge.score-100 { background: #e6f4ea; color: #2a9d3e; }
  .cell-header .score-badge.score-high { background: #eef6e1; color: #5a9e1a; }
  .cell-header .score-badge.score-mid { background: #fef3e0; color: #d4740a; }
  .cell-header .score-badge.score-low { background: #fde8e8; color: #c92a2a; }
  .model-link {
    color: inherit;
    text-decoration: none;
  }
  .model-link:hover {
    text-decoration: underline;
  }
  .cell-body {
    padding: 8px 12px;
    font-size: 13px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 32px;
    cursor: pointer;
    max-height: 80px;
    overflow: hidden;
  }
  .cell-body.expanded { max-height: none; }
  .error-badge { background: #fff0ef; color: #c92a2a; padding: 1px 6px; border: 1.5px solid #c92a2a; border-radius: 2px; font-size: 12px; }
  .time-col { width: 90px; padding: 10px 12px; font-size: 12px; color: #888; }
  .empty { text-align: center; padding: 80px 20px; color: #999; font-size: 16px; }
</style>
</head>
<body>

<div class="sidebar">
  <h1>smollest</h1>
  <h2>Projects</h2>
  <div id="project-list"></div>
</div>
<div class="main" id="main">
  <div class="empty">&#8592; pick a project over there</div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const projects = Object.keys(DATA);
const sidebar = document.getElementById('project-list');
const main = document.getElementById('main');

function formatCost(c) {
  if (c === null || c === undefined) return '\u2014';
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

function barColor(pct) {
  if (pct >= 80) return 'green';
  if (pct >= 50) return 'orange';
  return 'red';
}

function esc(s) { return (s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function isOpenModel(model) {
  return !(model.startsWith('http://') || model.startsWith('https://'));
}

function modelHeaderHtml(model, isBaseline = false) {
  const label = esc(model);
  if (isBaseline || !isOpenModel(model)) {
    return `<span>${label}</span>`;
  }
  const href = `https://huggingface.co/${encodeURI(model)}`;
  return `<a class="model-link" href="${href}" target="_blank" rel="noreferrer noopener">${label}</a>`;
}

function compareTag(val, baseVal, unit) {
  if (!baseVal || !val || baseVal === 0) return '';
  const ratio = val / baseVal;
  if (Math.abs(ratio - 1) < 0.05) return ' <span style="color:#888">(~same)</span>';
  if (ratio < 1) {
    const factor = baseVal / val;
    const label = factor >= 1.95 ? factor.toFixed(1) + 'x' : Math.round((1 - ratio) * 100) + '%';
    return ' <span style="color:#2a9d3e">(' + label + ' \u2193)</span>';
  }
  const factor = ratio;
  const label = factor >= 1.95 ? factor.toFixed(1) + 'x' : Math.round((ratio - 1) * 100) + '%';
  return ' <span style="color:#c92a2a">(' + label + ' \u2191)</span>';
}

function renderProject(name) {
  document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
  document.querySelector(`.sidebar a[data-project="${name}"]`)?.classList.add('active');

  const entries = DATA[name] || [];
  if (!entries.length) {
    main.innerHTML = '<div class="empty">nothing here yet. go run some comparisons!</div>';
    return;
  }

  const candStats = {};
  const queryMap = {};
  let totalQueries = 0;
  const baselineModel = entries[0]?.baseline_model || '?';
  const candOrder = [];

  entries.forEach(e => {
    if (!candStats[e.candidate]) {
      candStats[e.candidate] = { scores: [], latencies: [], costs: [] };
      candOrder.push(e.candidate);
    }
    if (e.score !== null) candStats[e.candidate].scores.push(e.score);
    candStats[e.candidate].latencies.push(e.candidate_latency_ms);
    if (e.candidate_cost !== null) candStats[e.candidate].costs.push(e.candidate_cost);

    const key = e.timestamp;
    if (!queryMap[key]) {
      queryMap[key] = { timestamp: e.timestamp, baseline_content: e.baseline_content, baseline_latency_ms: e.baseline_latency_ms, baseline_cost: e.baseline_cost, candidates: {} };
      totalQueries++;
    }
    queryMap[key].candidates[e.candidate] = e;
  });

  const baselineLatencies = Object.values(queryMap).map(q => q.baseline_latency_ms || 0);
  const avgBaselineLatency = baselineLatencies.reduce((a,b) => a+b, 0) / baselineLatencies.length;
  const totalBaselineCost = Object.values(queryMap).reduce((s, q) => s + (q.baseline_cost || 0), 0);

  let topRowHtml = `
    <div class="metric-card">
      <div class="label">Baseline</div>
      <div class="value blue">${baselineModel}</div>
      <div class="detail">${avgBaselineLatency.toFixed(0)}ms avg \u00b7 ${formatCost(totalBaselineCost)} total</div>
    </div>
    <div class="metric-card">
      <div class="label">Queries</div>
      <div class="value">${totalQueries}</div>
    </div>`;

  let simBars = '';
  let latBars = `
    <div class="bar-row">
      <div class="bar-label">${baselineModel}</div>
      <div class="bar-track"><div class="bar-fill baseline-fill" style="width:100%"></div></div>
      <div class="bar-val">${avgBaselineLatency.toFixed(0)}ms</div>
    </div>`;
  let costBars = `
    <div class="bar-row">
      <div class="bar-label">${baselineModel}</div>
      <div class="bar-track"><div class="bar-fill baseline-fill" style="width:100%"></div></div>
      <div class="bar-val">${formatCost(totalBaselineCost)}</div>
    </div>`;

  const allAvgLats = [avgBaselineLatency];
  const allTotalCosts = [totalBaselineCost];

  for (const cand of candOrder) {
    const d = candStats[cand];
    const avgLat = d.latencies.reduce((a,b)=>a+b,0) / d.latencies.length;
    const totalCost = d.costs.reduce((a,b)=>a+b,0);
    allAvgLats.push(avgLat);
    allTotalCosts.push(totalCost);
  }
  const maxLat = Math.max(...allAvgLats);
  const maxCost = Math.max(...allTotalCosts);

  const baseLatPct = maxLat > 0 ? (avgBaselineLatency / maxLat * 100) : 0;
  const baseCostPct = maxCost > 0 ? (totalBaselineCost / maxCost * 100) : 0;
  latBars = `
    <div class="bar-row">
      <div class="bar-label">${baselineModel}</div>
      <div class="bar-track"><div class="bar-fill baseline-fill" style="width:${baseLatPct}%"></div></div>
      <div class="bar-val">${avgBaselineLatency.toFixed(0)}ms</div>
    </div>`;
  costBars = `
    <div class="bar-row">
      <div class="bar-label">${baselineModel}</div>
      <div class="bar-track"><div class="bar-fill baseline-fill" style="width:${baseCostPct}%"></div></div>
      <div class="bar-val">${formatCost(totalBaselineCost)}</div>
    </div>`;

  simBars += `
    <div class="bar-row">
      <div class="bar-label">${baselineModel}</div>
      <div class="bar-track"><div class="bar-fill baseline-fill" style="width:100%"></div></div>
      <div class="bar-val">1.00</div>
    </div>`;

  for (const cand of candOrder) {
    const d = candStats[cand];
    const avgScore = d.scores.length ? d.scores.reduce((a,b)=>a+b,0) / d.scores.length : 0;
    const avgLat = d.latencies.reduce((a,b)=>a+b,0) / d.latencies.length;
    const totalCost = d.costs.reduce((a,b)=>a+b,0);
    const simPct = (avgScore * 100).toFixed(0);
    const latPct = maxLat > 0 ? (avgLat / maxLat * 100) : 0;
    const costPct = maxCost > 0 ? (totalCost / maxCost * 100) : 0;
    const shortName = cand.split('/').pop();

    simBars += `
      <div class="bar-row">
        <div class="bar-label" title="${esc(cand)}">${esc(shortName)}</div>
        <div class="bar-track"><div class="bar-fill ${barColor(Number(simPct))}" style="width:${simPct}%"></div></div>
        <div class="bar-val">${(avgScore).toFixed(2)}</div>
      </div>`;
    latBars += `
      <div class="bar-row">
        <div class="bar-label" title="${esc(cand)}">${esc(shortName)}</div>
        <div class="bar-track"><div class="bar-fill baseline-fill" style="width:${latPct}%"></div></div>
        <div class="bar-val">${avgLat.toFixed(0)}ms${compareTag(avgLat, avgBaselineLatency, 'ms')}</div>
      </div>`;
    costBars += `
      <div class="bar-row">
        <div class="bar-label" title="${esc(cand)}">${esc(shortName)}</div>
        <div class="bar-track"><div class="bar-fill baseline-fill" style="width:${costPct}%"></div></div>
        <div class="bar-val">${formatCost(totalCost)}${compareTag(totalCost, totalBaselineCost, '$')}</div>
      </div>`;
  }

  topRowHtml += `
    <div class="chart-card">
      <div class="chart-title">Similarity</div>
      ${simBars}
    </div>
    <div class="chart-card">
      <div class="chart-title">Avg Latency</div>
      ${latBars}
    </div>
    <div class="chart-card">
      <div class="chart-title">Total Cost</div>
      ${costBars}
    </div>`;

  const modelCols = [baselineModel, ...candOrder];
  let thHtml = '<th class="time-col">Time</th>';
  modelCols.forEach((m, idx) => {
    thHtml += `<th>${modelHeaderHtml(m, idx === 0)}</th>`;
  });

  let rowsHtml = '';
  const queries = Object.values(queryMap).sort((a,b) => b.timestamp.localeCompare(a.timestamp));

  queries.forEach(q => {
    const timeStr = new Date(q.timestamp).toLocaleString();
    let cells = `<td class="time-col">${timeStr}</td>`;

    cells += `<td>
      <div class="cell-header">
        <span>${(q.baseline_latency_ms||0).toFixed(0)}ms</span>
        <span>${formatCost(q.baseline_cost)}</span>
      </div>
      <div class="cell-body" onclick="this.classList.toggle('expanded')">${esc(q.baseline_content)}</div>
    </td>`;

    for (const cand of candOrder) {
      const e = q.candidates[cand];
      if (!e) {
        cells += '<td><div class="cell-body" style="color:#999">\u2014</div></td>';
        continue;
      }
      if (e.error) {
        cells += `<td>
          <div class="cell-header"><span class="error-badge">${esc(e.error)}</span></div>
          <div class="cell-body" style="color:#999">\u2014</div>
        </td>`;
        continue;
      }
      const sc = e.score !== null ? (e.score * 100).toFixed(0) + '%' : '\u2014';
      const scCls = scoreClass(e.score);
      cells += `<td>
        <div class="cell-header">
          <span class="score-badge ${scCls}">${sc} match</span>
          <span>${(e.candidate_latency_ms||0).toFixed(0)}ms</span>
          <span>${formatCost(e.candidate_cost)}</span>
        </div>
        <div class="cell-body" onclick="this.classList.toggle('expanded')">${esc(e.candidate_content || e.baseline_content || '')}</div>
      </td>`;
    }

    rowsHtml += `<tr>${cells}</tr>`;
  });

  main.innerHTML = `
    <h2>${name}</h2>
    <div class="subtitle">${totalQueries} queries \u00b7 baseline: ${baselineModel}</div>
    <div class="top-row">${topRowHtml}</div>
    <table>
      <thead><tr>${thHtml}</tr></thead>
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
    print(f"smollest dashboard: {url}")
    print("Press Ctrl+C to stop")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
