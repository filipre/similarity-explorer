"""Generate an interactive HTML visualization of word similarity.

Reads the two long-format similarity tables (nuance_similarity.csv and
accepted_answers_similarity.csv) plus vocab.csv, and writes a single
self-contained HTML file with a force-directed network graph. The page lets
you switch between the two similarity metrics, filter to a JLPT level, and
drag a threshold slider to control how strict a match counts.

Outputs:
  data/similarity_explorer.html
"""

import json

import pandas as pd

VOCAB_CSV = "data/vocab.csv"
NUANCE_SIMILARITY_CSV = "data/nuance_similarity.csv"
ACCEPTED_ANSWERS_SIMILARITY_CSV = "data/accepted_answers_similarity.csv"
OUTPUT_FILE = "data/similarity_explorer.html"

ACCEPTED_DISPLAY_LIMIT = 8


def build_edges(df: pd.DataFrame) -> list[dict]:
    """Collapse a long-format top-k neighbor table into deduped undirected edges.

    Each row is a directed (id -> similar_id) match; the same pair can appear
    from both directions with slightly different rank. Keep one edge per pair,
    taking the higher of the two scores (cosine similarity is symmetric, so
    they're normally equal).
    """
    best: dict[tuple[int, int], float] = {}
    for row in df.itertuples(index=False):
        a, b = int(row.id), int(row.similar_id)
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        score = round(float(row.similarity), 4)
        if key not in best or score > best[key]:
            best[key] = score
    return [{"source": a, "target": b, "score": s} for (a, b), s in best.items()]


def format_accepted(text: str, limit: int = ACCEPTED_DISPLAY_LIMIT) -> str:
    items = [a.strip() for a in str(text).split(",") if a.strip()]
    if len(items) > limit:
        return ", ".join(items[:limit]) + f", +{len(items) - limit} more"
    return ", ".join(items)


def build_nodes(vocab: pd.DataFrame) -> list[dict]:
    nodes = []
    for row in vocab.itertuples(index=False):
        nodes.append(
            {
                "id": int(row.id),
                "title": row.title,
                "meaning": row.meaning,
                "nuance": row.nuance_translation,
                "jlpt": row.jlpt_level,
                "accepted": format_accepted(row.accepted_answers),
            }
        )
    return nodes


def build_metrics(nuance_df: pd.DataFrame, accepted_df: pd.DataFrame) -> dict:
    nuance_top_k = int(nuance_df["rank"].max())
    accepted_top_k = int(accepted_df["rank"].max())
    return {
        "nuance": {
            "label": "Nuance similarity",
            "sliderMin": 0.55,
            "sliderMax": 1.0,
            "sliderStep": 0.005,
            "sliderDefault": 0.8,
            "methodHtml": (
                "Method: <code>BAAI/bge-large-en-v1.5</code> sentence embeddings over each word's "
                "<code>nuance_translation</code> text, normalized, compared by cosine similarity "
                f"(top {nuance_top_k} matches per word). The network layout is recomputed from "
                "every match whenever you change metric or level; the threshold slider only changes "
                "what is drawn."
            ),
            "edges": build_edges(nuance_df),
        },
        "accepted_answers": {
            "label": "Accepted answers similarity",
            "sliderMin": 0.0,
            "sliderMax": 1.0,
            "sliderStep": 0.01,
            "sliderDefault": 0.2,
            "methodHtml": (
                "Method: TF-IDF over each word's <code>accepted_answers</code>, tokenized so every "
                "comma-separated phrase (e.g. &ldquo;to move&rdquo;) counts as one feature instead of "
                "being split into individual words, compared by cosine similarity "
                f"(top {accepted_top_k} matches per word). Many pairs share no phrase at all and sit "
                "at 0.0 &mdash; raise the threshold to focus on real overlap."
            ),
            "edges": build_edges(accepted_df),
        },
    }


HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Similarity Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7;
    --surface: #fcfcfb;
    --surface-2: #f2f1ec;
    --ink: #0b0b0b;
    --ink-secondary: #52514e;
    --ink-muted: #898781;
    --hairline: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --accent: #2a78d6;
    --seq-1: #b7d3f6;
    --seq-2: #86b6ef;
    --seq-3: #5598e7;
    --seq-4: #2a78d6;
    --seq-5: #1c5cab;
    --seq-6: #104281;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d;
      --surface: #1a1a19;
      --surface-2: #212120;
      --ink: #ffffff;
      --ink-secondary: #c3c2b7;
      --ink-muted: #898781;
      --hairline: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --accent: #3987e5;
      --seq-1: #104281;
      --seq-2: #1c5cab;
      --seq-3: #2a78d6;
      --seq-4: #5598e7;
      --seq-5: #86b6ef;
      --seq-6: #b7d3f6;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --surface-2: #212120;
    --ink: #ffffff;
    --ink-secondary: #c3c2b7;
    --ink-muted: #898781;
    --hairline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
    --seq-1: #104281;
    --seq-2: #1c5cab;
    --seq-3: #2a78d6;
    --seq-4: #5598e7;
    --seq-5: #86b6ef;
    --seq-6: #b7d3f6;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page);
    color: var(--ink);
    font-family: 'IBM Plex Sans', 'Noto Sans JP', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .page { max-width: 1180px; margin: 0 auto; padding: 40px 24px 56px; }

  .hero { max-width: 680px; }
  .eyebrow {
    margin: 0 0 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-muted);
  }
  h1 {
    margin: 0 0 12px;
    font-size: clamp(28px, 4vw, 36px);
    font-weight: 700;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }
  .lede { margin: 0; font-size: 15px; line-height: 1.55; color: var(--ink-secondary); }
  .lede strong { color: var(--ink); font-weight: 600; }

  .kpis {
    margin-top: 28px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--hairline);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    overflow: hidden;
  }
  .tile { background: var(--surface); padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
  .tile-value {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  .tile-label { font-size: 11.5px; color: var(--ink-secondary); line-height: 1.35; }

  .panel { margin-top: 20px; background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px; overflow: hidden; }
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--hairline);
    flex-wrap: wrap;
  }
  .toolbar-left { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; flex: 1 1 auto; }
  .search-wrap { position: relative; flex: 1 1 220px; max-width: 320px; }
  .search-wrap input {
    width: 100%;
    padding: 8px 12px 8px 32px;
    border-radius: 8px;
    border: 1px solid var(--hairline);
    background: var(--surface-2);
    color: var(--ink);
    font-size: 13.5px;
    font-family: inherit;
  }
  .search-wrap input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .search-wrap svg { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--ink-muted); }

  .select-wrap { display: flex; align-items: center; gap: 8px; flex: 0 0 auto; }
  .select-wrap label { font-size: 11.5px; color: var(--ink-secondary); white-space: nowrap; }
  .select-wrap select {
    padding: 7px 10px;
    border-radius: 8px;
    border: 1px solid var(--hairline);
    background: var(--surface-2);
    color: var(--ink);
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
  }
  .select-wrap select:focus { outline: 2px solid var(--accent); outline-offset: -1px; }

  .threshold-wrap { display: flex; align-items: center; gap: 10px; flex: 1 1 220px; max-width: 300px; }
  .threshold-wrap label { font-size: 11.5px; color: var(--ink-secondary); white-space: nowrap; }
  .threshold-wrap input[type=range] {
    flex: 1 1 auto;
    accent-color: var(--accent);
    height: 4px;
  }
  .threshold-value {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    font-weight: 600;
    min-width: 3.4em;
    text-align: right;
  }

  .toolbar-right { display: flex; align-items: center; gap: 8px; flex: 0 0 auto; }

  .view-toggle { display: flex; gap: 2px; background: var(--surface-2); padding: 2px; border-radius: 8px; border: 1px solid var(--hairline); }
  .view-toggle button {
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 6px;
    border: none;
    background: transparent;
    color: var(--ink-secondary);
    cursor: pointer;
  }
  .view-toggle button.active { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px var(--border); }
  .view-toggle button:hover:not(.active) { color: var(--ink); }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    flex: 0 0 auto;
    border-radius: 8px;
    border: 1px solid var(--hairline);
    background: var(--surface-2);
    color: var(--ink-secondary);
    cursor: pointer;
  }
  .icon-btn:hover { color: var(--ink); background: var(--surface); }
  .icon-btn svg { width: 15px; height: 15px; }
  .icon-btn[aria-pressed="true"] { color: var(--accent); border-color: var(--accent); }

  .stage {
    position: relative;
    height: clamp(420px, 62vh, 640px);
    background:
      linear-gradient(var(--hairline) 1px, transparent 1px) 0 0 / 32px 32px,
      linear-gradient(90deg, var(--hairline) 1px, transparent 1px) 0 0 / 32px 32px,
      var(--surface);
    touch-action: none;
  }
  .stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; touch-action: none; }
  .stage canvas:active { cursor: grabbing; }

  /* Kept as separate rules (not a comma list): a selector list where any one
     selector is unrecognized invalidates the whole rule, and Firefox has no
     concept of the legacy :-webkit-full-screen pseudo-class. */
  .panel:fullscreen {
    display: flex;
    flex-direction: column;
    width: 100vw;
    height: 100vh;
    max-width: none;
    border: none;
    border-radius: 0;
    background: var(--page);
  }
  .panel:-webkit-full-screen {
    display: flex;
    flex-direction: column;
    width: 100vw;
    height: 100vh;
    max-width: none;
    border: none;
    border-radius: 0;
    background: var(--page);
  }
  .panel:fullscreen .stage {
    flex: 1 1 auto;
    height: auto;
  }
  .panel:-webkit-full-screen .stage {
    flex: 1 1 auto;
    height: auto;
  }
  .panel:fullscreen .list-wrap {
    flex: 1 1 auto;
    max-height: none;
  }
  .panel:-webkit-full-screen .list-wrap {
    flex: 1 1 auto;
    max-height: none;
  }

  .legend {
    position: absolute;
    right: 14px;
    bottom: 14px;
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    backdrop-filter: blur(4px);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 11px;
    color: var(--ink-secondary);
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 168px;
  }
  .legend-row { display: flex; flex-direction: column; gap: 5px; }
  .legend-title { font-weight: 600; color: var(--ink); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; }
  .legend-ramp { height: 8px; border-radius: 4px; background: linear-gradient(90deg, var(--seq-1), var(--seq-2), var(--seq-3), var(--seq-4), var(--seq-5), var(--seq-6)); }
  .legend-scale { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 10px; }
  .legend-size { display: flex; align-items: center; gap: 10px; }
  .legend-size .dots { display: flex; align-items: flex-end; gap: 6px; }
  .legend-size circle { fill: var(--ink-muted); }
  .zoom-hint {
    position: absolute;
    left: 14px;
    bottom: 14px;
    font-size: 10.5px;
    color: var(--ink-muted);
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    border: 1px solid var(--hairline);
    border-radius: 8px;
    padding: 5px 9px;
  }

  .tooltip {
    position: absolute;
    pointer-events: none;
    max-width: 260px;
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    box-shadow: 0 8px 24px var(--border);
    padding: 10px 12px;
    font-size: 12.5px;
    line-height: 1.4;
    z-index: 5;
  }
  .tooltip .t-title { font-size: 15px; font-weight: 700; }
  .tooltip .t-meaning { color: var(--ink-secondary); margin-top: 2px; }
  .tooltip .t-nuance { color: var(--ink-secondary); margin-top: 6px; font-size: 11.5px; }
  .tooltip .t-accepted { color: var(--ink-secondary); margin-top: 6px; font-size: 11px; }
  .tooltip .t-degree { margin-top: 6px; font-size: 11px; color: var(--ink-muted); }
  .tooltip .t-neighbors { margin-top: 8px; border-top: 1px solid var(--hairline); padding-top: 6px; display: flex; flex-direction: column; gap: 3px; }
  .tooltip .t-neighbor { display: flex; justify-content: space-between; gap: 8px; }
  .tooltip .t-neighbor .n-word { font-weight: 500; }
  .tooltip .t-neighbor .n-score { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; color: var(--accent); }
  .tooltip .t-neighbor.below { opacity: 0.45; }

  .empty-state {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--ink-muted);
    font-size: 13px;
    text-align: center;
    padding: 20px;
    pointer-events: none;
  }
  .empty-state[hidden] { display: none; }

  .list-wrap { max-height: clamp(420px, 62vh, 640px); overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    position: sticky;
    top: 0;
    background: var(--surface-2);
    text-align: left;
    padding: 9px 14px;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink-muted);
    border-bottom: 1px solid var(--hairline);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  thead th.sortable:hover { color: var(--ink); }
  thead th .arrow { margin-left: 4px; opacity: 0.6; }
  tbody td { padding: 9px 14px; border-bottom: 1px solid var(--hairline); vertical-align: top; }
  tbody tr:hover { background: var(--surface-2); }
  .cell-word { font-weight: 600; }
  .cell-meaning { color: var(--ink-secondary); font-size: 12px; margin-top: 1px; }
  .cell-score { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
  tbody tr.match-row { background: color-mix(in srgb, var(--accent) 10%, transparent); }

  .method { margin-top: 18px; font-size: 11.5px; color: var(--ink-muted); line-height: 1.5; max-width: 760px; }
  .method code { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }

  @media (max-width: 720px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .toolbar-left { flex-direction: column; align-items: stretch; }
  }
</style>

<div class="page">
  <header class="hero">
    <p class="eyebrow" id="eyebrow">JLPT vocabulary &middot; nuance similarity</p>
    <h1>Similarity Explorer</h1>
    <p class="lede">
      Two signals surface words that are easy to mix up: <strong>nuance similarity</strong> (how a word feels,
      from its English gloss) and <strong>accepted answers similarity</strong> (how much its literal English
      translations overlap with another word's). Switch metric, filter by JLPT level, and drag the threshold to
      control how strict a match counts.
    </p>
  </header>

  <section class="kpis" id="kpis"></section>

  <section class="panel">
    <div class="toolbar">
      <div class="toolbar-left">
        <div class="search-wrap">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
          <input id="search" type="text" placeholder="Search a word or meaning" autocomplete="off">
        </div>
        <div class="select-wrap">
          <label for="metric">Metric</label>
          <select id="metric">
            <option value="nuance" selected>Nuance similarity</option>
            <option value="accepted_answers">Accepted answers similarity</option>
          </select>
        </div>
        <div class="select-wrap">
          <label for="level">Level</label>
          <select id="level">
            <option value="all" selected>All levels</option>
            <option value="N4">N4</option>
            <option value="N5">N5</option>
          </select>
        </div>
        <div class="threshold-wrap">
          <label for="threshold">Threshold</label>
          <input id="threshold" type="range" min="0.55" max="1" step="0.005" value="0.8">
          <span class="threshold-value" id="thresholdValue">0.800</span>
        </div>
      </div>
      <div class="toolbar-right">
        <div class="view-toggle" role="tablist">
          <button data-view="network" class="active" role="tab" aria-selected="true">Network</button>
          <button data-view="list" role="tab" aria-selected="false">List</button>
        </div>
        <button id="fullscreenBtn" class="icon-btn" type="button" title="Fullscreen" aria-pressed="false">
          <svg class="icon-expand" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>
        </button>
      </div>
    </div>

    <div class="stage" id="stage">
      <canvas id="canvas"></canvas>
      <div class="legend">
        <div class="legend-row">
          <span class="legend-title">Similarity</span>
          <div class="legend-ramp"></div>
          <div class="legend-scale"><span id="legendMin">0.80</span><span>1.00</span></div>
        </div>
        <div class="legend-row">
          <span class="legend-title">Word&nbsp;&mdash;&nbsp;matches</span>
          <div class="legend-size">
            <svg width="44" height="28" class="dots">
              <circle cx="8" cy="20" r="4"></circle>
              <circle cx="24" cy="16" r="8"></circle>
              <circle cx="40" cy="10" r="13"></circle>
            </svg>
            <span>few &rarr; many</span>
          </div>
        </div>
      </div>
      <div class="zoom-hint">Scroll or pinch to zoom &middot; drag to pan</div>
      <div class="tooltip" id="tooltip" hidden></div>
      <div class="empty-state" id="emptyState" hidden></div>
    </div>

    <div class="list-wrap" id="listWrap" hidden>
      <table id="table">
        <thead>
          <tr>
            <th>Word</th>
            <th>Similar word</th>
            <th class="sortable" data-sort="score">Similarity <span class="arrow">&darr;</span></th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </section>

  <p class="method" id="method"></p>
</div>

<script id="graph-data" type="application/json">__GRAPH_DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
(function () {
  "use strict";

  var raw = JSON.parse(document.getElementById('graph-data').textContent);

  var metricSelect = document.getElementById('metric');
  var levelSelect = document.getElementById('level');
  var thresholdInput = document.getElementById('threshold');
  var thresholdValueEl = document.getElementById('thresholdValue');
  var legendMinEl = document.getElementById('legendMin');
  var eyebrowEl = document.getElementById('eyebrow');
  var methodEl = document.getElementById('method');
  var searchInput = document.getElementById('search');

  var SLIDER_MIN = 0;
  var threshold = 0;
  var nodes = [];
  var nodesById = new Map();
  var edges = [];
  var adjacency = new Map();
  var visibleEdges = [];
  var visibleDegree = new Map();
  var visibleAdjacency = new Map();
  var simulation = null;
  var lastMetric = null;

  // ---- rebuild everything (nodes, edges, layout) on metric or level change;
  // the threshold slider then only toggles what is drawn, without resimulating ----
  function rebuild() {
    var metric = metricSelect.value;
    var level = levelSelect.value;
    var cfg = raw.metrics[metric];

    nodes = level === 'all' ? raw.nodes.slice() : raw.nodes.filter(function (n) { return n.jlpt === level; });
    nodes.forEach(function (n) {
      n.x = (Math.random() - 0.5) * 1400;
      n.y = (Math.random() - 0.5) * 1400;
    });
    nodesById = new Map();
    nodes.forEach(function (n) { nodesById.set(n.id, n); });

    edges = cfg.edges
      .filter(function (e) { return nodesById.has(e.source) && nodesById.has(e.target); })
      .map(function (e) { return { source: e.source, target: e.target, score: e.score }; });

    adjacency = new Map();
    nodes.forEach(function (n) { adjacency.set(n.id, []); });
    edges.forEach(function (e) {
      adjacency.get(e.source).push({ id: e.target, score: e.score });
      adjacency.get(e.target).push({ id: e.source, score: e.score });
    });
    adjacency.forEach(function (list) { list.sort(function (a, b) { return b.score - a.score; }); });

    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(function (d) { return d.id; })
        .distance(function (l) { return 20 + (1 - l.score) * 200; })
        .strength(function (l) { return 0.05 + l.score * 0.4; }))
      .force('charge', d3.forceManyBody().strength(-24).distanceMax(500))
      .force('collide', d3.forceCollide(6))
      .force('center', d3.forceCenter(0, 0))
      .force('x', d3.forceX(0).strength(0.02))
      .force('y', d3.forceY(0).strength(0.02))
      .stop();
    for (var i = 0; i < 400; i++) simulation.tick();

    SLIDER_MIN = cfg.sliderMin;
    thresholdInput.min = cfg.sliderMin;
    thresholdInput.max = cfg.sliderMax;
    thresholdInput.step = cfg.sliderStep;
    if (lastMetric !== metric) thresholdInput.value = cfg.sliderDefault;
    lastMetric = metric;
    threshold = parseFloat(thresholdInput.value);
    thresholdValueEl.textContent = threshold.toFixed(3);
    legendMinEl.textContent = threshold.toFixed(2);

    eyebrowEl.textContent = 'JLPT vocabulary · ' + cfg.label.toLowerCase() +
      (level === 'all' ? '' : ' · ' + level + ' only');
    methodEl.innerHTML = cfg.methodHtml;

    hovered = null;
    pinned = null;

    recomputeThreshold();
    renderKpis();
    fitToView();
    applySearch(searchInput.value);
    draw();
  }

  function recomputeThreshold() {
    visibleEdges = edges.filter(function (e) { return e.score >= threshold; });
    visibleDegree = new Map();
    visibleAdjacency = new Map();
    nodes.forEach(function (n) { visibleAdjacency.set(n.id, []); });
    visibleEdges.forEach(function (e) {
      var sId = e.source.id, tId = e.target.id;
      visibleDegree.set(sId, (visibleDegree.get(sId) || 0) + 1);
      visibleDegree.set(tId, (visibleDegree.get(tId) || 0) + 1);
      visibleAdjacency.get(sId).push(tId);
      visibleAdjacency.get(tId).push(sId);
    });
  }

  // ---- KPIs (recomputed on every threshold/metric/level change) ----
  var kpiEl = document.getElementById('kpis');
  function renderKpis() {
    var visited = new Set();
    var largest = 0;
    visibleDegree.forEach(function (_, id) {
      if (visited.has(id)) return;
      var size = 0, queue = [id];
      visited.add(id);
      while (queue.length) {
        var cur = queue.pop();
        size++;
        visibleAdjacency.get(cur).forEach(function (nb) {
          if (!visited.has(nb)) { visited.add(nb); queue.push(nb); }
        });
      }
      if (size > largest) largest = size;
    });
    var meanScore = visibleEdges.length
      ? visibleEdges.reduce(function (s, e) { return s + e.score; }, 0) / visibleEdges.length
      : 0;

    var tiles = [
      [visibleEdges.length.toLocaleString(), 'confusable pairs at this threshold'],
      [visibleDegree.size.toLocaleString(), 'of ' + nodes.length.toLocaleString() + ' words involved'],
      [largest.toLocaleString(), 'words in the largest cluster'],
      [visibleEdges.length ? meanScore.toFixed(3) : '–', 'mean similarity of matched pairs']
    ];
    kpiEl.innerHTML = '';
    tiles.forEach(function (t) {
      var tile = document.createElement('div');
      tile.className = 'tile';
      var v = document.createElement('span'); v.className = 'tile-value'; v.textContent = t[0];
      var l = document.createElement('span'); l.className = 'tile-label'; l.textContent = t[1];
      tile.appendChild(v); tile.appendChild(l);
      kpiEl.appendChild(tile);
    });
  }

  // ---- color helpers ----
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function hexToRgb(hex) {
    var h = hex.replace('#', '');
    if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var num = parseInt(h, 16);
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
  }
  var seqCache = null;
  function seqStops() {
    seqCache = ['--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5', '--seq-6'].map(function (v) { return hexToRgb(cssVar(v)); });
    return seqCache;
  }
  function seqColor(t, alpha) {
    var stops = seqCache || seqStops();
    t = Math.max(0, Math.min(1, t));
    var pos = t * (stops.length - 1);
    var i = Math.min(stops.length - 2, Math.floor(pos));
    var f = pos - i;
    var a = stops[i], b = stops[i + 1];
    var r = a[0] + (b[0] - a[0]) * f;
    var g = a[1] + (b[1] - a[1]) * f;
    var bl = a[2] + (b[2] - a[2]) * f;
    return 'rgba(' + r.toFixed(0) + ',' + g.toFixed(0) + ',' + bl.toFixed(0) + ',' + alpha + ')';
  }

  // ---- canvas + hand-rolled pan/zoom (no dependency on a DOM zoom library) ----
  var stage = document.getElementById('stage');
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  var tooltip = document.getElementById('tooltip');
  var emptyState = document.getElementById('emptyState');
  var dpr = Math.max(1, window.devicePixelRatio || 1);
  var transform = { x: 0, y: 0, k: 1 };
  var MIN_K = 0.08, MAX_K = 10;
  var hovered = null;
  var pinned = null;
  var matchedIds = null;

  function resize() {
    var rect = stage.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    draw();
  }

  function fitToView() {
    if (!nodes.length) return;
    var xs = nodes.map(function (n) { return n.x; });
    var ys = nodes.map(function (n) { return n.y; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var rect = stage.getBoundingClientRect();
    var w = maxX - minX || 1, h = maxY - minY || 1;
    var scale = Math.min(rect.width / (w * 1.1), rect.height / (h * 1.1), 2.2);
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    transform.k = Math.max(MIN_K, Math.min(MAX_K, scale));
    transform.x = rect.width / 2 - cx * transform.k;
    transform.y = rect.height / 2 - cy * transform.k;
  }

  function zoomAt(px, py, factor) {
    var newK = Math.max(MIN_K, Math.min(MAX_K, transform.k * factor));
    var gx = (px - transform.x) / transform.k;
    var gy = (py - transform.y) / transform.k;
    transform.k = newK;
    transform.x = px - gx * newK;
    transform.y = py - gy * newK;
  }

  canvas.addEventListener('wheel', function (ev) {
    ev.preventDefault();
    var rect = canvas.getBoundingClientRect();
    var px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    var factor = Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.03 : 0.0016));
    zoomAt(px, py, factor);
    draw();
  }, { passive: false });

  // Pointer Events unify mouse/touch/pen: single pointer pans, two pointers pinch-zoom.
  var pointers = new Map();
  var dragMoved = false;
  var pinchStartDist = null;

  function pointerPos(ev) {
    var rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  }
  function midpoint(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  canvas.addEventListener('pointerdown', function (ev) {
    canvas.setPointerCapture(ev.pointerId);
    pointers.set(ev.pointerId, pointerPos(ev));
    dragMoved = false;
    pinchStartDist = null;
  });
  canvas.addEventListener('pointermove', function (ev) {
    if (!pointers.has(ev.pointerId)) { handleHover(ev); return; }
    var prev = pointers.get(ev.pointerId);
    var cur = pointerPos(ev);
    pointers.set(ev.pointerId, cur);

    if (pointers.size === 1) {
      var dx = cur.x - prev.x, dy = cur.y - prev.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
      transform.x += dx;
      transform.y += dy;
      hideTooltip();
      draw();
    } else if (pointers.size === 2) {
      var pts = Array.from(pointers.values());
      var d = dist(pts[0], pts[1]);
      var mid = midpoint(pts[0], pts[1]);
      if (pinchStartDist != null) {
        zoomAt(mid.x, mid.y, d / pinchStartDist);
        dragMoved = true;
        draw();
      }
      pinchStartDist = d;
    }
  });
  function endPointer(ev) {
    pointers.delete(ev.pointerId);
    if (pointers.size < 2) pinchStartDist = null;
    if (!dragMoved && pointers.size === 0) handleClick(ev);
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', function (ev) { pointers.delete(ev.pointerId); });
  canvas.addEventListener('pointerleave', function (ev) {
    if (!pointers.has(ev.pointerId)) { hovered = null; hideTooltip(); draw(); }
  });

  function nodeAlpha(n) {
    var visible = visibleDegree.has(n.id);
    if (matchedIds) return matchedIds.has(n.id) ? (visible ? 1 : 0.35) : 0.06;
    var focus = pinned || hovered;
    if (!focus) return visible ? 0.92 : 0.16;
    if (n.id === focus.id) return 1;
    var isNeighbor = adjacency.get(focus.id).some(function (a) { return a.id === n.id; });
    if (isNeighbor) return visible ? 0.95 : 0.3;
    return 0.05;
  }
  function edgeAlphaFactor(e) {
    if (matchedIds) return (matchedIds.has(e.source.id) && matchedIds.has(e.target.id)) ? 1 : 0.04;
    var focus = pinned || hovered;
    if (!focus) return 1;
    var touches = e.source.id === focus.id || e.target.id === focus.id;
    return touches ? 1 : 0.03;
  }

  function draw() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    var scoreSpan = Math.max(0.001, 1 - SLIDER_MIN);
    visibleEdges.forEach(function (e) {
      var t = Math.max(0, Math.min(1, (e.score - SLIDER_MIN) / scoreSpan));
      var a = edgeAlphaFactor(e) * (0.2 + t * 0.55);
      ctx.strokeStyle = seqColor(t, a);
      ctx.lineWidth = (0.5 + t * 1.8) / Math.max(1, transform.k * 0.6);
      ctx.beginPath();
      ctx.moveTo(e.source.x, e.source.y);
      ctx.lineTo(e.target.x, e.target.y);
      ctx.stroke();
    });

    var muted = cssVar('--ink-muted');
    var ink = cssVar('--ink');
    var accent = cssVar('--accent');
    var focus = pinned || hovered;
    var maxVisibleDegree = 1;
    visibleDegree.forEach(function (d) { if (d > maxVisibleDegree) maxVisibleDegree = d; });

    nodes.forEach(function (n) {
      var deg = visibleDegree.get(n.id) || 0;
      var r = 4 + Math.sqrt(deg) * 2.6;
      var t = deg <= 1 ? 0 : Math.min(1, Math.log2(deg) / Math.log2(Math.max(2, maxVisibleDegree)));
      var alpha = nodeAlpha(n);
      var isFocus = focus && n.id === focus.id;
      var isMatched = matchedIds && matchedIds.has(n.id);
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = seqColor(t, alpha);
      ctx.fill();
      if (isFocus || isMatched) {
        ctx.lineWidth = 1.6 / transform.k;
        ctx.strokeStyle = accent;
        ctx.stroke();
      }

      var showLabel = deg >= 8 || isFocus || isMatched;
      if (showLabel && transform.k > 0.28) {
        ctx.font = '500 ' + (11.5 / Math.max(0.7, Math.min(1.4, transform.k))) + 'px "Noto Sans JP","IBM Plex Sans",sans-serif';
        ctx.fillStyle = isFocus || isMatched ? ink : muted;
        ctx.globalAlpha = Math.max(alpha, isFocus || isMatched ? 1 : 0.6);
        ctx.textBaseline = 'middle';
        ctx.fillText(n.title, n.x + r + 4 / transform.k, n.y);
        ctx.globalAlpha = 1;
      }
    });

    ctx.restore();
  }

  window.addEventListener('resize', resize);

  function nearestNode(gx, gy) {
    var best = null, bestDist = Infinity;
    nodes.forEach(function (n) {
      var deg = visibleDegree.get(n.id) || 0;
      var r = 4 + Math.sqrt(deg) * 2.6;
      var dx = n.x - gx, dy = n.y - gy;
      var d = dx * dx + dy * dy;
      var hit = Math.max(r + 6, 10);
      if (d < hit * hit && d < bestDist) { best = n; bestDist = d; }
    });
    return best;
  }
  function graphPoint(px, py) {
    return { x: (px - transform.x) / transform.k, y: (py - transform.y) / transform.k };
  }
  function handleHover(ev) {
    var p = pointerPos(ev);
    var g = graphPoint(p.x, p.y);
    var n = nearestNode(g.x, g.y);
    if (n !== hovered) { hovered = n; draw(); }
    if (n) { showTooltip(n, p.x, p.y); canvas.style.cursor = 'pointer'; }
    else { hideTooltip(); canvas.style.cursor = 'grab'; }
  }
  function handleClick(ev) {
    var p = pointerPos(ev);
    var g = graphPoint(p.x, p.y);
    var n = nearestNode(g.x, g.y);
    pinned = (pinned && n && pinned.id === n.id) ? null : n;
    draw();
  }

  function showTooltip(n, px, py) {
    tooltip.hidden = false;
    tooltip.innerHTML = '';
    var title = document.createElement('div'); title.className = 't-title'; title.textContent = n.title;
    var meaning = document.createElement('div'); meaning.className = 't-meaning'; meaning.textContent = n.meaning;
    var nuance = document.createElement('div'); nuance.className = 't-nuance'; nuance.textContent = n.nuance;
    var accepted = document.createElement('div'); accepted.className = 't-accepted'; accepted.textContent = 'Accepted: ' + n.accepted;
    var degree = document.createElement('div'); degree.className = 't-degree';
    var deg = visibleDegree.get(n.id) || 0;
    degree.textContent = 'Confusable with ' + deg + ' word' + (deg === 1 ? '' : 's') + ' at this threshold';
    tooltip.appendChild(title); tooltip.appendChild(meaning); tooltip.appendChild(nuance);
    tooltip.appendChild(accepted); tooltip.appendChild(degree);

    var top3 = adjacency.get(n.id).slice(0, 3);
    if (top3.length) {
      var box = document.createElement('div'); box.className = 't-neighbors';
      top3.forEach(function (nb) {
        var row = document.createElement('div');
        row.className = 't-neighbor' + (nb.score < threshold ? ' below' : '');
        var w = document.createElement('span'); w.className = 'n-word'; w.textContent = nodesById.get(nb.id).title;
        var s = document.createElement('span'); s.className = 'n-score'; s.textContent = nb.score.toFixed(3);
        row.appendChild(w); row.appendChild(s);
        box.appendChild(row);
      });
      tooltip.appendChild(box);
    }

    var stageRect = stage.getBoundingClientRect();
    var tw = 260, pad = 12;
    var left = px + 16, top = py + 16;
    if (left + tw > stageRect.width) left = px - tw - 16;
    if (top > stageRect.height - 200) top = Math.max(pad, py - 200);
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
  }
  function hideTooltip() { tooltip.hidden = true; }

  // ---- threshold slider ----
  function onThresholdChange() {
    threshold = parseFloat(thresholdInput.value);
    thresholdValueEl.textContent = threshold.toFixed(3);
    legendMinEl.textContent = threshold.toFixed(2);
    recomputeThreshold();
    renderKpis();
    draw();
    renderTable();
  }
  thresholdInput.addEventListener('input', onThresholdChange);

  // ---- metric / level selects ----
  metricSelect.addEventListener('change', rebuild);
  levelSelect.addEventListener('change', rebuild);

  // ---- search ----
  function applySearch(q) {
    q = q.trim().toLowerCase();
    if (!q) { matchedIds = null; emptyState.hidden = true; draw(); renderTable(); return; }
    var ids = new Set();
    nodes.forEach(function (n) {
      if (n.title.toLowerCase().indexOf(q) !== -1 || n.meaning.toLowerCase().indexOf(q) !== -1) ids.add(n.id);
    });
    matchedIds = ids;
    emptyState.hidden = ids.size > 0;
    if (!emptyState.hidden) emptyState.textContent = 'No words match "' + searchInput.value.trim() + '"';
    draw();
    renderTable();
  }
  searchInput.addEventListener('input', function () { applySearch(searchInput.value); });
  searchInput.addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter' && matchedIds && matchedIds.size) {
      var firstId = matchedIds.values().next().value;
      var n = nodesById.get(firstId);
      var rect = stage.getBoundingClientRect();
      transform.k = Math.max(transform.k, 1.4);
      transform.x = rect.width / 2 - n.x * transform.k;
      transform.y = rect.height / 2 - n.y * transform.k;
      draw();
    }
  });

  // ---- view toggle ----
  var buttons = document.querySelectorAll('.view-toggle button');
  var stageEl = document.getElementById('stage');
  var listWrap = document.getElementById('listWrap');
  buttons.forEach(function (btn) {
    btn.addEventListener('click', function () {
      buttons.forEach(function (b) { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
      btn.classList.add('active'); btn.setAttribute('aria-selected', 'true');
      var isNetwork = btn.dataset.view === 'network';
      stageEl.hidden = !isNetwork;
      listWrap.hidden = isNetwork;
      if (!isNetwork) renderTable();
    });
  });

  // ---- fullscreen ----
  var panelEl = document.querySelector('.panel');
  var fullscreenBtn = document.getElementById('fullscreenBtn');
  var iconExpand = fullscreenBtn.querySelector('.icon-expand');
  var iconCompress = fullscreenBtn.querySelector('.icon-compress');

  function fullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }
  if (!document.fullscreenEnabled && !document.webkitFullscreenEnabled) {
    fullscreenBtn.hidden = true;
  }
  fullscreenBtn.addEventListener('click', function () {
    if (fullscreenElement()) {
      (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    } else {
      var request = panelEl.requestFullscreen || panelEl.webkitRequestFullscreen;
      request.call(panelEl);
    }
  });
  function onFullscreenChange() {
    var active = fullscreenElement() === panelEl;
    fullscreenBtn.title = active ? 'Exit fullscreen' : 'Fullscreen';
    fullscreenBtn.setAttribute('aria-pressed', active ? 'true' : 'false');
    iconExpand.hidden = active;
    iconCompress.hidden = !active;
    // Entering/exiting fullscreen triggers a CSS-driven layout change (.stage
    // growing to fill the screen) that isn't necessarily done yet when this
    // event fires. Measuring stage size too early re-centers against the old,
    // smaller box. Wait a couple of frames so layout has actually settled.
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        resize();
        fitToView();
        draw();
      });
    });
  }
  document.addEventListener('fullscreenchange', onFullscreenChange);
  document.addEventListener('webkitfullscreenchange', onFullscreenChange);

  // ---- list view ----
  var sortDir = -1;
  var tableBody = document.getElementById('tableBody');
  function renderTable() {
    var rows = visibleEdges.slice().sort(function (a, b) { return (b.score - a.score) * sortDir * -1; });
    var q = searchInput.value.trim().toLowerCase();
    tableBody.innerHTML = '';
    var frag = document.createDocumentFragment();
    rows.forEach(function (e) {
      var a = nodesById.get(e.source.id !== undefined ? e.source.id : e.source);
      var b = nodesById.get(e.target.id !== undefined ? e.target.id : e.target);
      var isMatch = q && (a.title.toLowerCase().indexOf(q) !== -1 || a.meaning.toLowerCase().indexOf(q) !== -1 ||
        b.title.toLowerCase().indexOf(q) !== -1 || b.meaning.toLowerCase().indexOf(q) !== -1);
      if (q && !isMatch) return;
      var tr = document.createElement('tr');
      if (isMatch) tr.className = 'match-row';

      var tdA = document.createElement('td');
      var wa = document.createElement('div'); wa.className = 'cell-word'; wa.textContent = a.title;
      var ma = document.createElement('div'); ma.className = 'cell-meaning'; ma.textContent = a.meaning;
      tdA.appendChild(wa); tdA.appendChild(ma);

      var tdB = document.createElement('td');
      var wb = document.createElement('div'); wb.className = 'cell-word'; wb.textContent = b.title;
      var mb = document.createElement('div'); mb.className = 'cell-meaning'; mb.textContent = b.meaning;
      tdB.appendChild(wb); tdB.appendChild(mb);

      var tdS = document.createElement('td'); tdS.className = 'cell-score'; tdS.textContent = e.score.toFixed(4);

      tr.appendChild(tdA); tr.appendChild(tdB); tr.appendChild(tdS);
      frag.appendChild(tr);
    });
    tableBody.appendChild(frag);
  }
  document.querySelector('th.sortable').addEventListener('click', function () {
    sortDir *= -1;
    this.querySelector('.arrow').textContent = sortDir === -1 ? '↓' : '↑';
    renderTable();
  });

  function onThemeChange() { seqStops(); draw(); }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', onThemeChange);
  new MutationObserver(onThemeChange).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---- boot ----
  seqStops();
  resize();
  rebuild();
})();
</script>
"""


def main() -> None:
    vocab = pd.read_csv(VOCAB_CSV)
    text_cols = ["title", "meaning", "nuance_translation", "jlpt_level", "accepted_answers"]
    vocab[text_cols] = vocab[text_cols].fillna("")

    nuance_df = pd.read_csv(NUANCE_SIMILARITY_CSV)
    accepted_df = pd.read_csv(ACCEPTED_ANSWERS_SIMILARITY_CSV)

    raw = {
        "nodes": build_nodes(vocab),
        "metrics": build_metrics(nuance_df, accepted_df),
    }

    payload = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</script", "<\\/script")

    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", payload)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {len(vocab)} words and {len(raw['metrics'])} metrics to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
