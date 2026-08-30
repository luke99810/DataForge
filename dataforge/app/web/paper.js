/* 论文链路的界面逻辑：上传 → 变量卡确认 → 回填续跑。
 *
 * 续跑不是单独的「恢复」流程：确认结果作为 confirm 节点的参数重新提交，
 * 上游（解析、抽取）指纹不变自动命中缓存，只有 confirm 及其下游重算。
 * 对论文链路这点很实在——extract 是要花钱的那一步。
 */

const P = { paperId: null, paperName: null, cards: [], questions: [], adopt: new Set() };

/* ── 模型状态 ─────────────────────────────────────── */
async function checkLLM() {
  const pill = $("#llmPill");
  try {
    const j = await (await fetch("/api/llm")).json();
    if (j.configured) {
      pill.textContent = j.model;
      pill.className = "pill g";
    } else {
      pill.textContent = "未配置模型";
      pill.className = "pill r";
      pill.title = j.hint || "";
    }
  } catch {
    pill.textContent = "状态未知";
    pill.className = "pill m";
  }
}

/* ── 上传 ─────────────────────────────────────────── */
$("#btnPick").onclick = () => $("#filePdf").click();
$("#filePdf").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("#paperName").textContent = "上传中…";
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await fetch("/api/papers", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "上传失败");
    P.paperId = j.paper_id;
    P.paperName = j.filename;
    $("#paperName").textContent =
      `${j.filename} · ${(j.size / 1024 / 1024).toFixed(1)} MB · ${j.paper_id}`;
    // 换论文就清掉上一篇的卡片，否则会拿旧卡去确认新论文
    P.cards = []; P.adopt.clear();
    $("#cardsBox").style.display = "none";
  } catch (err) {
    $("#paperName").textContent = "上传失败";
    $("#errBox").innerHTML = `<div class="err-box">${esc(err.message)}</div>`;
  }
};

/* ── 变量卡 ───────────────────────────────────────── */
function renderCards(ctx) {
  P.cards = ctx.variables || [];
  P.questions = ctx.open_questions || [];
  P.adopt = new Set(ctx.suggested || []);
  $("#cardsBox").style.display = "";

  const box = $("#cards");
  box.innerHTML = "";
  P.cards.forEach((c) => {
    const conf = Number(c.confidence || 0);
    const bad = (c.semantic_issues || []).length > 0;
    const related = !!c.from_related_work;
    const wrap = el("div", "vcard" + (related || bad ? " low" : ""));

    const span = c.source_span || {};
    const range = [c.min, c.max].every((v) => v == null) ? "—" : `${c.min ?? "—"} – ${c.max ?? "—"}`;
    const confColor = conf >= 0.8 ? "var(--green)" : conf >= 0.5 ? "var(--amber)" : "var(--red)";

    wrap.innerHTML = `
      <div class="vc-h">
        <label class="vc-pick">
          <input type="checkbox" ${P.adopt.has(c.name) ? "checked" : ""} data-n="${esc(c.name)}">
        </label>
        <span class="vc-n">${esc(c.name)}</span>
        <span class="pill ${c.role === "target" ? "p" : "b"}">${esc(c.role || "input")}</span>
        ${related ? '<span class="pill a">来自相关工作</span>' : ""}
        ${bad ? '<span class="pill r">语义问题</span>' : ""}
        <span style="flex:1"></span>
        <span class="conf"><span class="cbar"><i style="width:${Math.round(conf * 100)}%;
          background:${confColor}"></i></span>${conf.toFixed(2)}</span>
      </div>
      <div class="vc-g">
        <div class="vc-f"><span class="k">类型</span><span class="v">${esc(c.dtype)}</span></div>
        <div class="vc-f"><span class="k">单位</span><span class="v">${esc(c.unit || "—")}</span></div>
        <div class="vc-f"><span class="k">范围</span><span class="v">${esc(range)}</span></div>
        <div class="vc-f"><span class="k">页码</span><span class="v">p.${esc(span.page ?? "?")}</span></div>
      </div>
      ${bad ? `<div class="vc-warn">${(c.semantic_issues || []).map(esc).join("；")}</div>` : ""}
      ${related ? `<div class="vc-warn">这句出现在<b>相关工作</b>而非本文方法。
        采纳它会生成一份论文本身并不包含的数据，复现结果就对不上了。</div>` : ""}
      <div class="vc-src"><span class="pill m">${esc(span.section || "?")}</span>
        <q>${esc((span.quote || "").slice(0, 180))}</q></div>`;

    wrap.querySelector("input").onchange = (e) => {
      if (e.target.checked) P.adopt.add(c.name); else P.adopt.delete(c.name);
      updateAdoptCount();
    };
    box.appendChild(wrap);
  });

  const q = $("#questions");
  q.innerHTML = P.questions.length
    ? `<div class="vc-q"><b>论文没写清楚的地方（不替你猜）</b>` +
      P.questions.map((x, i) => `
        <div class="qrow">
          <div class="qt">${esc(x.question)}</div>
          ${x.why ? `<div class="qw">${esc(x.why)}</div>` : ""}
          <input class="qa" data-i="${i}" placeholder="${
            esc((x.options || []).join(" / ") || "你的决定")}">
        </div>`).join("") + "</div>"
    : "";
  updateAdoptCount();
}

function updateAdoptCount() {
  $("#btnConfirm").textContent = `确认并继续（${P.adopt.size}）`;
  $("#btnConfirm").disabled = P.adopt.size === 0;
}

$("#btnAdoptSuggested").onclick = () => {
  P.cards.forEach((c) => (c.suggest_adopt ? P.adopt.add(c.name) : P.adopt.delete(c.name)));
  document.querySelectorAll("#cards input[type=checkbox]").forEach((cb) => {
    cb.checked = P.adopt.has(cb.dataset.n);
  });
  updateAdoptCount();
};

$("#btnConfirm").onclick = () => {
  const answers = {};
  document.querySelectorAll("#questions .qa").forEach((inp) => {
    const i = +inp.dataset.i;
    if (inp.value.trim()) answers[P.questions[i].question] = inp.value.trim();
  });
  runPaper({ adopt: [...P.adopt], answers });
};

/* ── 论文流水线的运行入口 ─────────────────────────── */
async function runPaper(decisions) {
  if (!P.paperId) {
    $("#errBox").innerHTML = `<div class="err-box">请先选择一个 PDF</div>`;
    return;
  }
  await runWith({
    kind: "paper",
    paper_id: P.paperId,
    name: $("#fName").value || "paper_dataset",
    num_records: +$("#fRows").value || 500,
    seed: +$("#fSeed").value || 0,
    fields: [],
    decisions: decisions || null,
    thresholds: collectThresholds(),
    known_correlated: [],
    force: [],
  });
}

/* 流水线切换时显示/隐藏对应区块 */
const origKindChange = $("#fKind").onchange;
$("#fKind").onchange = (e) => {
  const paper = $("#fKind").value === "paper";
  $("#paperBox").style.display = paper ? "" : "none";
  $("#fldBoxWrap").style.display = paper ? "none" : "";
  if (paper) { checkLLM(); } else { $("#cardsBox").style.display = "none"; }
  if (origKindChange) origKindChange.call($("#fKind"), e);
};
