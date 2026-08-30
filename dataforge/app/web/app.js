/* DataForge 桌面界面。
 *
 * 无构建步骤：整个前端就是三个静态文件。桌面版要打成 PyInstaller 单文件，
 * 少一条 npm 构建链就少一处会在打包时炸掉的地方。等 UI 复杂到值得上
 * React/Vite 时再换（设计文档 §10 的选型可以随时切回去）。
 */
const $ = (s) => document.querySelector(s);
const el = (t, cls, txt) => {
  const n = document.createElement(t);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = {
  runId: null, since: 0, timer: null, dag: null,
  results: {}, artifacts: {}, quality: null, analyze: null, dataRef: null,
};

/* ── 导航 ─────────────────────────────────────────── */
$("#rail").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  document.querySelectorAll("#rail button").forEach((x) => x.classList.remove("on"));
  b.classList.add("on");
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("on"));
  $("#" + b.dataset.s).classList.add("on");
  if (b.dataset.s === "s5") loadHistory();
});

/* ── 字段编辑 ─────────────────────────────────────── */
const TYPES = ["float", "integer", "string", "categorical", "datetime", "boolean"];
let FIELDS = [
  { name: "sensor_id", type: "string" },
  { name: "speed", type: "float", min: 0, max: 70, extra: "mph" },
  { name: "volume", type: "integer", min: 0, max: 1200, extra: "veh/5min" },
  { name: "occupancy", type: "float", min: 0, max: 1 },
  { name: "level", type: "categorical", extra: "low,mid,high" },
];

function renderFields() {
  const box = $("#fields");
  box.innerHTML = "";
  FIELDS.forEach((f, i) => {
    const row = el("div", "grid4");
    const name = el("input"); name.value = f.name; name.placeholder = "字段名";
    name.oninput = () => { f.name = name.value; refreshSelectors(); };

    const type = el("select");
    TYPES.forEach((t) => { const o = el("option", null, t); o.value = t; type.appendChild(o); });
    type.value = f.type;
    type.onchange = () => { f.type = type.value; renderFields(); refreshSelectors(); };

    const mn = el("input"); mn.type = "number"; mn.placeholder = "—";
    mn.value = f.min ?? ""; mn.oninput = () => { f.min = mn.value === "" ? null : +mn.value; };
    const mx = el("input"); mx.type = "number"; mx.placeholder = "—";
    mx.value = f.max ?? ""; mx.oninput = () => { f.max = mx.value === "" ? null : +mx.value; };

    const ex = el("input");
    ex.placeholder = f.type === "categorical" ? "a,b,c"
      : f.type === "datetime" ? "2026-01-01,2026-03-01" : "单位";
    ex.value = f.extra ?? ""; ex.oninput = () => { f.extra = ex.value; };

    const del = el("button", "iconbtn", "×");
    del.onclick = () => { FIELDS.splice(i, 1); renderFields(); refreshSelectors(); };

    const numeric = f.type === "float" || f.type === "integer";
    mn.disabled = mx.disabled = !numeric;
    if (!numeric) { mn.style.opacity = mx.style.opacity = ".35"; }

    [name, type, mn, mx, ex, del].forEach((n) => row.appendChild(n));
    box.appendChild(row);
  });
  $("#fldCount").textContent = FIELDS.length;
}

$("#btnAddField").onclick = () => {
  FIELDS.push({ name: "field_" + (FIELDS.length + 1), type: "float" });
  renderFields(); refreshSelectors();
};

function refreshSelectors() {
  const fill = (sel, pred, allowNone) => {
    const cur = sel.value;
    sel.innerHTML = "";
    if (allowNone) { const o = el("option", null, "（不指定）"); o.value = ""; sel.appendChild(o); }
    FIELDS.filter(pred).forEach((f) => {
      const o = el("option", null, f.name); o.value = f.name; sel.appendChild(o);
    });
    if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  };
  fill($("#fTarget"), () => true, true);
  fill($("#fTime"), (f) => f.type === "datetime", false);
}

$("#fKind").onchange = () => {
  const ts = $("#fKind").value === "timeseries";
  $("#wrapTime").style.display = ts ? "" : "none";
  if (ts && !FIELDS.some((f) => f.type === "datetime")) {
    FIELDS.unshift({ name: "ts", type: "datetime", extra: "2026-01-01,2026-03-01" });
    renderFields();
  }
  refreshSelectors();
};

/* ── 组装请求 ─────────────────────────────────────── */
function buildRequest(force) {
  const fields = FIELDS.filter((f) => f.name.trim()).map((f) => {
    const o = { name: f.name.trim(), type: f.type };
    if (f.min != null && f.min !== "") o.min = +f.min;
    if (f.max != null && f.max !== "") o.max = +f.max;
    if (f.extra) {
      if (f.type === "categorical") o.categories = f.extra.split(",").map((s) => s.trim()).filter(Boolean);
      else if (f.type === "datetime") {
        const [a, b] = f.extra.split(",").map((s) => s.trim());
        if (a) o.start = a; if (b) o.end = b;
      } else o.unit = f.extra;
    }
    return o;
  });
  return {
    kind: $("#fKind").value,
    name: $("#fName").value || "dataset",
    fields,
    num_records: +$("#fRows").value || 100,
    seed: +$("#fSeed").value || 0,
    target: $("#fTarget").value || null,
    time_column: $("#fKind").value === "timeseries" ? ($("#fTime").value || null) : null,
    thresholds: collectThresholds(),
    known_correlated: [],
    force: force ? (S.dag ? S.dag.nodes.map((n) => n.id) : []) : [],
    auto_refine: $("#fAuto").checked,
    max_rounds: +$("#fRounds").value || 5,
  };
}

function collectThresholds() {
  const th = {};
  if ($("#tRows").value) th.min_rows = +$("#tRows").value;
  if ($("#tComp").value) th.min_completeness = +$("#tComp").value;
  return th;
}

/* ── 运行 ─────────────────────────────────────────── */
$("#btnRun").onclick = () => run(false);
$("#btnForce").onclick = () => run(true);
$("#btnClearTrace").onclick = () => { $("#trace").innerHTML = ""; };

async function run(force) {
  if ($("#fKind").value === "paper") { runPaper(null); return; }
  await runWith(buildRequest(force));
}

async function runWith(body) {
  $("#errBox").innerHTML = "";
  $("#askBox").innerHTML = "";
  $("#btnRun").disabled = $("#btnForce").disabled = true;
  $("#trace").innerHTML = "";
  S.since = 0; S.results = {}; S.quality = null; S.analyze = null; S.dataRef = null;
  seenSeq.clear();
  $("#loopBox").innerHTML = "";

  try {
    const r = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "启动失败");
    S.runId = j.run_id;
    S.dag = j.dag;
    $("#crumb").textContent = "— " + j.run_id;
    $("#tracePill").textContent = "运行中";
    $("#tracePill").className = "pill b";
    drawGraph();
    poll();
  } catch (e) {
    $("#errBox").innerHTML = `<div class="err-box">${esc(e.message)}</div>`;
    $("#btnRun").disabled = $("#btnForce").disabled = false;
  }
}

function poll() {
  clearTimeout(S.timer);
  S.timer = setTimeout(async () => {
    try {
      const r = await fetch(`/api/runs/${S.runId}?since=${S.since}`);
      const j = await r.json();
      S.since = j.next_since;
      S.results = j.results || {};
      j.events.forEach(addTrace);
      drawGraph();
      $("#stNodes").textContent = Object.keys(S.results).length || "—";

      if (j.status === "running" || j.status === "pending") { poll(); return; }
      finish(j);
    } catch (e) {
      $("#errBox").innerHTML = `<div class="err-box">轮询失败：${esc(e.message)}</div>`;
      $("#btnRun").disabled = $("#btnForce").disabled = false;
    }
  }, 350);
}

const STATUS_TEXT = {
  success: ["全部通过", "g"], blocked: ["被门禁阻断", "r"],
  failed: ["执行失败", "r"], needs_human: ["需要人工确认", "a"], error: ["内部错误", "r"],
};

async function finish(j) {
  $("#btnRun").disabled = $("#btnForce").disabled = false;
  const [txt, cls] = STATUS_TEXT[j.status] || [j.status, "m"];
  $("#tracePill").textContent = txt;
  $("#tracePill").className = "pill " + cls;
  $("#stStatus").innerHTML = `<span class="pill ${cls}">${txt}</span>`;

  const cached = Object.values(S.results).filter((r) => r.status === "CACHED").length;
  const total = Object.keys(S.results).length;
  $("#runHint").textContent = `${total} 个节点，命中缓存 ${cached}`;

  if (j.error) $("#errBox").innerHTML = `<div class="err-box">${esc(j.error)}</div>`;
  if (j.ask) {
    $("#askBox").innerHTML =
      `<div class="ask-box"><b>需要人工确认</b><br>${esc(j.ask.question || "")}
       <div class="muted" style="margin-top:6px;font-size:11.5px">
       这一步不替你猜。中间产物已全部保留，确认后只会重跑受影响的分支。</div></div>`;
    const ctx = j.ask.context || {};
    if (ctx.variables && typeof renderCards === "function") renderCards(ctx);
  }

  if (j.loop) renderLoop(j.loop);
  await loadArtifacts();
  refreshHealth();
}

/* ── 轨迹 ─────────────────────────────────────────── */
const KIND_META = {
  node_start: ["START", "start"], node_end: ["DONE", "end"],
  node_cached: ["CACHED", "cached"], node_error: ["ERROR", "err"],
  gate: ["GATE", "gate"], ask: ["ASK", "gate"], note: ["NOTE", "note"],
  run_start: ["RUN", "start"], run_end: ["RUN", "end"],
};

const seenSeq = new Set();
function addTrace(e) {
  // 轮询窗口重叠时同一事件可能到两次，按 seq 去重
  if (e.seq != null) {
    if (seenSeq.has(e.seq)) return;
    seenSeq.add(e.seq);
  }
  const box = $("#trace");
  if (box.querySelector(".empty")) box.innerHTML = "";
  const [label, cls] = KIND_META[e.kind] || [e.kind.toUpperCase(), "note"];
  const row = el("div", "tr" + (e.kind === "node_start" ? " act" : ""));

  const h = el("div", "tr-h");
  h.appendChild(el("span", "tr-k " + cls, label));
  h.appendChild(el("span", "tr-t", new Date().toLocaleTimeString("zh-CN", { hour12: false })));
  row.appendChild(h);

  const b = el("div", "tr-b");
  if (e.node_id) {
    b.innerHTML = `<code>${esc(e.node_id)}</code>` +
      (e.kind === "node_cached" ? " 命中缓存，跳过计算" : "") +
      (e.duration != null ? ` · ${e.duration}s` : "");
  } else if (e.kind === "run_start") {
    b.textContent = `开始执行 ${e.nodes ?? ""} 个节点`;
  } else if (e.kind === "run_end") {
    b.textContent = `结束：${e.status ?? ""}`;
  } else {
    b.textContent = e.message || e.msg || "";
  }
  row.appendChild(b);

  if (e.message || e.msg) {
    const r = el("div", "tr-r" + (e.drift ? " bad" : ""));
    r.textContent = e.message || e.msg;
    row.appendChild(r);
  }
  if (e.error) {
    const r = el("div", "tr-r bad");
    r.textContent = e.error;
    row.appendChild(r);
  }
  if (e.fingerprint) {
    const r = el("div", "tr-r");
    r.textContent = "fp " + e.fingerprint.slice(0, 16) + "…";
    row.appendChild(r);
  }

  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

/* ── 图谱 ─────────────────────────────────────────── */
const NODE_COLOR = {
  COMPUTED: "#3fb950", CACHED: "#bc8cff", BLOCKED: "#f85149",
  FAILED: "#f85149", SKIPPED: "#6e7681",
};

function drawGraph() {
  if (!S.dag) return;
  $("#canvasEmpty").style.display = "none";
  const old = $("#canvas svg");
  if (old) old.remove();

  // 按拓扑层级排布：入度为 0 的在第一层，其余取上游最大层 + 1
  const nodes = S.dag.nodes, edges = S.dag.edges;
  const parents = {}; nodes.forEach((n) => (parents[n.id] = []));
  edges.forEach((e) => parents[e.to] && parents[e.to].push(e.from));

  const layer = {};
  const depth = (id, seen = new Set()) => {
    if (layer[id] != null) return layer[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = parents[id] || [];
    const d = ps.length ? Math.max(...ps.map((p) => depth(p, seen))) + 1 : 0;
    layer[id] = d;
    return d;
  };
  nodes.forEach((n) => depth(n.id));

  const byLayer = {};
  nodes.forEach((n) => (byLayer[layer[n.id]] ||= []).push(n));
  const W = 168, H = 66, GX = 62, GY = 34;
  const pos = {};
  Object.keys(byLayer).sort((a, b) => a - b).forEach((L) => {
    byLayer[L].forEach((n, i) => {
      pos[n.id] = { x: 30 + L * (W + GX), y: 30 + i * (H + GY) };
    });
  });
  const maxL = Math.max(...Object.values(layer));
  const maxR = Math.max(...Object.values(byLayer).map((a) => a.length));

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("width", 30 + (maxL + 1) * (W + GX));
  svg.setAttribute("height", 60 + maxR * (H + GY));
  svg.style.display = "block";

  const defs = document.createElementNS(NS, "defs");
  defs.innerHTML = `<marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5"
      orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#3d4650"/></marker>`;
  svg.appendChild(defs);

  edges.forEach((e) => {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) return;
    const p = document.createElementNS(NS, "path");
    const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
    p.setAttribute("d", `M${x1},${y1} C${x1 + 34},${y1} ${x2 - 34},${y2} ${x2},${y2}`);
    p.setAttribute("stroke", "#3d4650");
    p.setAttribute("stroke-width", "1.6");
    p.setAttribute("fill", "none");
    p.setAttribute("marker-end", "url(#ar)");
    svg.appendChild(p);
  });

  nodes.forEach((n) => {
    const p = pos[n.id], r = S.results[n.id];
    const color = r ? (NODE_COLOR[r.status] || "#58a6ff") : "#30363d";
    const g = document.createElementNS(NS, "g");
    g.setAttribute("transform", `translate(${p.x},${p.y})`);
    const dash = r ? "" : ' stroke-dasharray="4 3"';
    g.innerHTML = `
      <rect width="${W}" height="${H}" rx="9" fill="#161b22"
            stroke="${color}" stroke-width="${r ? 1.8 : 1.4}"${dash}/>
      <text x="13" y="21" fill="#6e7681" font-size="9.5"
            font-family="monospace">${esc(n.kind)}</text>
      <text x="13" y="39" fill="#e6edf3" font-size="12.5" font-weight="600">${esc(n.id)}</text>
      <text x="13" y="54" fill="#6e7681" font-size="10">${
        r ? esc(r.status) + (r.duration ? ` · ${r.duration}s` : "") : "等待"}</text>
      <circle cx="${W - 13}" cy="18" r="4" fill="${color}"/>`;
    g.style.cursor = "pointer";
    g.onclick = () => showNode(n.id);
    svg.appendChild(g);
  });

  $("#canvas").appendChild(svg);
}

function showNode(id) {
  const r = S.results[id];
  if (!r) return;
  alertPanel(`${id}\n状态 ${r.status}\n指纹 ${r.fingerprint}\n` +
    (r.artifact ? `产物 ${r.artifact.kind}:${r.artifact.hash}` : "") +
    (r.error ? `\n错误 ${r.error}` : ""));
}
function alertPanel(text) {
  // 不用 window.alert —— 模态框会卡死 WebView2 的消息循环
  const box = $("#errBox");
  box.innerHTML = `<div class="ask-box" style="white-space:pre-wrap">${esc(text)}</div>`;
  document.querySelector('#rail button[data-s="s1"]').click();
  box.scrollIntoView({ behavior: "smooth" });
}

/* ── 质量与数据 ───────────────────────────────────── */
async function loadArtifacts() {
  const get = async (nid) => {
    const r = S.results[nid];
    if (!r || !r.artifact) return null;
    const res = await fetch(`/api/artifacts/${r.artifact.kind}/${r.artifact.hash}`);
    return res.ok ? res.json() : null;
  };
  const gate = await get("gate");
  S.quality = gate ? gate.value : null;
  const an = await get("analyze");
  S.analyze = an ? an.value : null;
  renderQuality();

  const g = S.results["generate"];
  if (g && g.artifact) {
    const d = await fetch(`/api/artifacts/dataframe/${g.artifact.hash}?limit=200`);
    if (d.ok) renderData(await d.json());
  }
}

function renderQuality() {
  const box = $("#quality");
  if (!S.quality) {
    box.innerHTML = `<div class="empty">这次运行没有产出质量报告${
      S.results.gate && S.results.gate.error ? "：" + esc(S.results.gate.error) : "。"}</div>`;
    return;
  }
  const q = S.quality, a = S.analyze || {};
  const audit = q.audit || {};
  const comp = (q.results || []).find((r) => r.gate === "completeness");
  const leak = (q.results || []).find((r) => r.gate === "leakage");
  const findings = (leak && leak.details && leak.details.findings) || [];

  const cards = `
    <div class="qgrid">
      <div class="qc"><div class="k">门禁总体</div>
        <div class="v" style="color:${q.passed ? "var(--green)" : "var(--red)"}">${q.passed ? "通过" : "未通过"}</div>
        <div class="d">${(q.results || []).length} 项检查</div></div>
      <div class="qc"><div class="k">完整性</div>
        <div class="v">${comp ? (comp.details.completeness * 100).toFixed(1) + "%" : "—"}</div>
        <div class="d">下限 ${comp ? (comp.details.min * 100).toFixed(0) + "%" : "—"}</div></div>
      <div class="qc"><div class="k">泄漏检测</div>
        <div class="v" style="color:${leak && leak.passed ? "var(--green)" : "var(--red)"}">${
          leak ? (leak.passed ? "未检出" : "检出") : "—"}</div>
        <div class="d">${findings.length} 条发现</div></div>
      <div class="qc"><div class="k">阈值</div>
        <div class="v" style="color:${audit.relaxed ? "var(--amber)" : "var(--fg)"}">${audit.relaxed ? "已放宽" : "默认"}</div>
        <div class="d">${audit.relaxed ? "已写进数据卡" : "未改动"}</div></div>
    </div>`;

  const issues = (q.results || []).map((r) => {
    const cls = r.passed ? "ok" : r.level === "BLOCK" ? "err" : r.level === "ASK" ? "wrn" : "wrn";
    const pill = r.passed ? "g" : r.level === "BLOCK" ? "r" : "a";
    return `<div class="issue ${cls}"><span class="pill ${pill}">${r.passed ? "PASS" : r.level}</span>
      <div style="flex:1"><div class="it">${esc(r.gate)}</div>
      <div class="id">${esc(r.message)}</div></div></div>`;
  }).join("");

  const leakList = findings.length ? `<h2 class="sec" style="margin-top:18px">泄漏检查明细</h2>
    <p class="sub">物理规律导致的高相关会被标为 info，不算泄漏</p>` +
    findings.map((f) => `<div class="issue ${
      f.severity === "error" ? "err" : f.severity === "warning" ? "wrn" : "inf"}">
      <span class="pill ${f.severity === "error" ? "r" : f.severity === "warning" ? "a" : "b"}">${
        f.severity.toUpperCase()}</span>
      <div style="flex:1"><div class="it">${esc(f.kind)} · ${esc((f.columns || []).join(" ↔ "))}</div>
      <div class="id">${esc(f.message)}</div></div></div>`).join("") : "";

  box.innerHTML = `<h2 class="sec">质量报告</h2>
    <p class="sub">${esc(S.runId || "")}</p>${cards}
    <h2 class="sec">门禁检查</h2>
    <p class="sub">任一 BLOCK 未通过则阻断导出</p>${issues}${leakList}
    ${a.histograms ? renderHistograms(a) : ""}`;
}

function renderHistograms(a) {
  const keys = Object.keys(a.histograms).slice(0, 4);
  if (!keys.length) return "";
  const charts = keys.map((k) => {
    const h = a.histograms[k];
    const max = Math.max(...h.counts, 1);
    const bw = 360 / h.counts.length;
    const bars = h.counts.map((c, i) => {
      const bh = (c / max) * 100;
      return `<rect x="${20 + i * bw}" y="${118 - bh}" width="${Math.max(1, bw - 2)}"
        height="${bh}" rx="2" fill="#58a6ff" opacity=".82"/>`;
    }).join("");
    const d = a.describe[k] || {};
    return `<div class="ch"><div class="ch-h">${esc(k)}</div>
      <div class="ch-s">均值 ${d.mean != null ? d.mean.toFixed(2) : "—"} ·
        中位 ${d.p50 != null ? d.p50.toFixed(2) : "—"} ·
        范围 ${d.min != null ? d.min.toFixed(1) : "—"}–${d.max != null ? d.max.toFixed(1) : "—"}</div>
      <svg viewBox="0 0 400 140" style="width:100%;height:auto">
        <line x1="20" y1="118" x2="392" y2="118" stroke="#21262d"/>${bars}</svg></div>`;
  }).join("");
  return `<h2 class="sec" style="margin-top:18px">分布</h2>
    <p class="sub">单峰/双峰形状对不对，看图比看分数快</p>
    <div class="two">${charts}</div>`;
}

function renderData(d) {
  $("#dataSub").textContent =
    `${d.rows} 行 × ${d.cols} 列${d.truncated ? `（只显示前 ${d.preview.length} 行）` : ""}`;
  const t = el("table", "dt");
  const thead = el("thead"), hr = el("tr");
  d.columns.forEach((c) => {
    const th = el("th");
    th.innerHTML = `${esc(c)}<br><span style="font-weight:400;opacity:.6">${esc(d.dtypes[c] || "")}</span>`;
    hr.appendChild(th);
  });
  thead.appendChild(hr); t.appendChild(thead);
  const tb = el("tbody");
  d.preview.forEach((row) => {
    const tr = el("tr");
    d.columns.forEach((c) => {
      const v = row[c];
      const td = el("td", null, v === null || v === undefined ? "—" : String(v));
      if (v === null || v === undefined) td.style.color = "var(--fg-mute)";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  $("#dataTable").innerHTML = "";
  $("#dataTable").appendChild(t);
}

/* ── 历史 ─────────────────────────────────────────── */
async function loadHistory() {
  const box = $("#history");
  try {
    const j = await (await fetch("/api/runs?limit=40")).json();
    if (!j.runs.length) { box.innerHTML = `<div class="empty">还没有运行记录。</div>`; return; }
    box.innerHTML = `<h2 class="sec">运行历史</h2>
      <p class="sub">缓存跨运行复用 —— 命中数越高说明重复计算越少</p>
      <div class="card" style="overflow:auto"><table class="dt">
      <thead><tr><th>run_id</th><th>状态</th><th>节点</th><th>命中缓存</th><th>标签</th></tr></thead>
      <tbody>${j.runs.map((r) => `<tr>
        <td>${esc(r.run_id)}</td>
        <td><span class="pill ${
          r.status === "success" ? "g" : r.status === "running" ? "b" : "r"}">${esc(r.status)}</span></td>
        <td>${r.nodes}</td><td>${r.cached}</td><td>${esc(r.label || "")}</td>
      </tr>`).join("")}</tbody></table></div>`;
  } catch (e) {
    box.innerHTML = `<div class="err-box">${esc(e.message)}</div>`;
  }
}

async function refreshHealth() {
  try {
    const j = await (await fetch("/api/health")).json();
    $("#stCache").textContent = `${j.cache.entries} 条 / 命中 ${j.cache.total_hits}`;
  } catch { /* 服务还没起来，忽略 */ }
}

/* ── 启动 ─────────────────────────────────────────── */
renderFields();
refreshSelectors();
$("#fTarget").value = "level";
refreshHealth();


/* ── 自动调参的逐轮记录 ───────────────────────────── */
const LOOP_REASON = {
  success: ["门禁全绿", "g"],
  plateau: ["停滞——已达当前方法上限", "a"],
  blocked: ["调参救不了，已停", "r"],
  exhausted: ["预算耗尽", "a"],
  needs_human: ["等待人工确认", "a"],
};

function renderLoop(loop) {
  const [txt, cls] = LOOP_REASON[loop.reason] || [loop.reason, "m"];
  const rows = loop.rounds.map((r, i) => {
    const best = i === loop.best_index;
    const changes = (r.proposal && r.proposal.changes || [])
      .map((c) => `${esc(c.param)} ${c.from}→${c.to}`).join("、");
    const why = (r.proposal && r.proposal.give_up_reason) || "";
    return `<div class="lrow${best ? " best" : ""}">
      <span class="pill ${best ? "g" : "m"}">第 ${r.index} 轮</span>
      <span class="lscore">${r.score.toFixed(1)}</span>
      <div class="lbody">
        <div class="lstatus">${esc(r.status)}${best ? "  · 最佳" : ""}</div>
        ${changes ? `<div class="lchg">调整：${changes}</div>` : ""}
        ${why ? `<div class="lwhy">${esc(why)}</div>` : ""}
      </div>
    </div>`;
  }).join("");

  $("#loopBox").innerHTML = `<div class="fldbox">
    <h3>自动调参结果 <span class="pill ${cls}">${esc(txt)}</span></h3>
    <div class="muted" style="font-size:11.5px;margin-bottom:9px">${esc(loop.message || "")}</div>
    ${rows}
    <div class="muted" style="font-size:11px;margin-top:8px">
      分数曲线：${loop.scores.map((s) => s.toFixed(1)).join(" → ")}
    </div>
  </div>`;
}


/* ── Planner：自然语言 → 填表单 ─────────────────── */
$("#btnPlan").onclick = async () => {
  const text = $("#fNl").value.trim();
  if (!text) { $("#errBox").innerHTML = `<div class="err-box">先描述一下需求</div>`; return; }
  const btn = $("#btnPlan");
  btn.disabled = true; btn.textContent = "规划中…";
  $("#errBox").innerHTML = ""; $("#planOut").innerHTML = "";
  try {
    const r = await fetch("/api/plan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, seed: +$("#fSeed").value || 20260823 }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "规划失败");
    applyPlan(j.plan, j.request);
    if (!j.request) {
      $("#planPill").textContent = "计划有问题";
      $("#planPill").className = "pill r";
    }
  } catch (e) {
    $("#errBox").innerHTML = `<div class="err-box">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "让 Agent 规划";
  }
};

function applyPlan(plan, req) {
  // req 为 null = 计划书有语义问题，只展示问题、不填表单：
  // 填一份自己都知道跑不通的表单，比不填更让人困惑
  if (req) fillFormFromPlan(req);
  renderPlanNotes(plan);
}

function fillFormFromPlan(req) {
  $("#fName").value = req.name || "dataset";
  $("#fKind").value = req.kind;
  $("#fRows").value = req.num_records;
  FIELDS = req.fields.map((f) => ({
    name: f.name, type: f.type,
    min: f.min ?? null, max: f.max ?? null,
    extra: f.categories ? f.categories.join(",")
      : (f.start ? [f.start, f.end].filter(Boolean).join(",") : (f.unit || "")),
  }));
  renderFields();
  refreshSelectors();
  if (req.target) $("#fTarget").value = req.target;
  $("#wrapTime").style.display = req.kind === "timeseries" ? "" : "none";
  if (req.time_column) $("#fTime").value = req.time_column;
  $("#paperBox").style.display = "none";
  $("#fldBoxWrap").style.display = "";
}

function renderPlanNotes(plan) {
  const s = plan.spec || {};
  const parts = [];
  if (plan.cached) parts.push(`<div class="prat">这段需求之前规划过，直接复用了结果（没有再花钱调模型）。</div>`);
  if (s.rationale) parts.push(`<div class="prat"><b>规划依据：</b>${esc(s.rationale)}</div>`);
  if (plan.fixes && plan.fixes.length)
    parts.push(`<div class="pfix"><b>已自动修正：</b>${plan.fixes.map(esc).join("；")}</div>`);
  if (plan.issues && plan.issues.length)
    parts.push(`<div class="pissue"><b>还有问题：</b>${plan.issues.map(esc).join("；")}</div>`);
  if (s.open_questions && s.open_questions.length)
    parts.push(`<div class="pissue" style="border-left-color:var(--amber);color:var(--fg-dim)">
      <b>没说清的地方（没替你做主，按默认值填了）：</b><br>${
        s.open_questions.map((q) => esc(q.question) +
          (q.options && q.options.length ? `（${q.options.map(esc).join(" / ")}）` : "")).join("<br>")}</div>`);
  $("#planOut").innerHTML = parts.join("");
  $("#planPill").textContent = plan.cached ? "已复用" : "已规划";
  $("#planPill").className = "pill g";
}
