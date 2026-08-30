"""M3 论文链路：解析 → 抽取 → 人工确认 → Schema → 数据集。

★ 本文件最重要的一条是 ``test_resume_after_confirmation_reuses_cache``：
续跑不是单独实现的「恢复」逻辑，而是指纹缓存的自然结果。
"""

from __future__ import annotations

import json

import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.graph.executor import Executor
from dataforge.llm.client import LLMError, StubLLM, _extract_json
from dataforge.llm.contracts import (
    ContractError, EXTRACT_VARIABLES, check_variable_semantics, validate_against,
)
from dataforge.pipelines import paper_pipeline

pytest.importorskip("fitz", reason="解析 PDF 需要 PyMuPDF")


# ── 造一篇真 PDF ──────────────────────────────────────────

PAPER_TEXT = [
    ("Spatio-Temporal Graph Networks for Traffic Forecasting\n"
     "Anonymous Authors\n\n"
     "Abstract\n"
     "We propose a model for short-term traffic prediction.\n"),
    ("2 Related Work\n"
     "Prior work incorporates weather features such as rainfall into the model,\n"
     "and some studies use driver demographics.\n"),
    ("4.1 Datasets\n"
     "We evaluate on 207 loop detectors aggregated at 5-minute intervals over 12 weeks.\n"
     "Each record contains traffic speed, volume and occupancy.\n"
     "Missing values were imputed using linear interpolation.\n\n"
     "4.2 Experimental Setup\n"
     "We split the data 70% / 10% / 20% chronologically.\n"),
]


@pytest.fixture()
def pdf(tmp_path):
    import fitz

    doc = fitz.open()
    for body in PAPER_TEXT:
        page = doc.new_page()
        page.insert_text((60, 80), body, fontsize=10)
    p = tmp_path / "paper.pdf"
    doc.save(p)
    doc.close()
    return p


EXTRACTION = {
    "task_type": "time_series_forecast",
    "dataset_hint": {"num_records": 1000, "time_granularity": "5min"},
    "open_questions": [
        {"question": "论文未给出原始缺失率，用多少？",
         "options": ["3%", "5%", "8%"], "why": "会直接影响复现出来的精度"}
    ],
    "variables": [
        {"name": "speed", "dtype": "float", "unit": "mph", "min": 0, "max": 70,
         "role": "target", "confidence": 0.94,
         "source_span": {"page": 3, "section": "4.1 Datasets",
                         "quote": "Each record contains traffic speed, volume and occupancy"}},
        {"name": "volume", "dtype": "integer", "unit": "veh/5min", "min": 0, "max": 1200,
         "role": "input", "confidence": 0.9,
         "source_span": {"page": 3, "section": "4.1 Datasets",
                         "quote": "Each record contains traffic speed, volume and occupancy"}},
        {"name": "occupancy", "dtype": "float", "min": 0, "max": 1,
         "role": "input", "confidence": 0.88,
         "source_span": {"page": 3, "section": "4.1 Datasets",
                         "quote": "Each record contains traffic speed, volume and occupancy"}},
        # 来自相关工作 —— 必须默认不采纳
        {"name": "rainfall", "dtype": "float", "min": 0, "max": 100,
         "role": "input", "confidence": 0.35, "from_related_work": True,
         "source_span": {"page": 2, "section": "2 Related Work",
                         "quote": "Prior work incorporates weather features such as rainfall"}},
    ],
}


@pytest.fixture()
def env(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    rs = RunStore(tmp_path / "runs.sqlite")
    llm = StubLLM(default=EXTRACTION)
    yield Executor(store, rs, workdir=tmp_path, llm=llm), store, llm
    rs.close()


# ── 契约校验 ──────────────────────────────────────────────

def test_contract_accepts_valid_extraction():
    validate_against(EXTRACTION, EXTRACT_VARIABLES)


def test_contract_rejects_missing_source_span():
    bad = json.loads(json.dumps(EXTRACTION))
    del bad["variables"][0]["source_span"]
    with pytest.raises(ContractError, match="source_span"):
        validate_against(bad, EXTRACT_VARIABLES)


def test_contract_rejects_bad_dtype():
    bad = json.loads(json.dumps(EXTRACTION))
    bad["variables"][0]["dtype"] = "matrix"
    with pytest.raises(ContractError, match="enum|取值必须"):
        validate_against(bad, EXTRACT_VARIABLES)


def test_contract_rejects_out_of_range_confidence():
    bad = json.loads(json.dumps(EXTRACTION))
    bad["variables"][0]["confidence"] = 1.7
    with pytest.raises(ContractError, match="不得大于"):
        validate_against(bad, EXTRACT_VARIABLES)


def test_semantic_check_catches_bad_range():
    issues = check_variable_semantics(
        {"name": "x", "dtype": "float", "min": 10, "max": 1,
         "source_span": {"quote": "a sufficiently long quote"}}
    )
    assert any("min" in i for i in issues)


def test_semantic_check_requires_usable_quote():
    issues = check_variable_semantics(
        {"name": "x", "dtype": "float", "source_span": {"quote": "短"}}
    )
    assert any("原文引用太短" in i for i in issues)


def test_semantic_check_requires_categories():
    issues = check_variable_semantics(
        {"name": "lvl", "dtype": "categorical",
         "source_span": {"quote": "a sufficiently long quote"}}
    )
    assert any("categories" in i for i in issues)


# ── JSON 抽取的健壮性 ─────────────────────────────────────

@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '好的，结果如下：\n```\n{"a": 1}\n```\n希望有帮助',
    '这是结果 {"a": 1} 完毕',
])
def test_extract_json_survives_wrappers(raw):
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_fails_loudly():
    with pytest.raises(LLMError):
        _extract_json("完全没有 JSON 的一段话")


def test_structured_retries_then_succeeds():
    """契约失败要把错误喂回去重试，而不是把脏数据传下去。"""
    llm = StubLLM(responses=[("第二次", {"variables": []})], default={"nope": 1})
    schema = {"type": "object", "required": ["variables"],
              "properties": {"variables": {"type": "array", "items": {"type": "object"}}}}
    with pytest.raises(LLMError, match="重试"):
        llm.structured(system="s", user="u", schema=schema, max_retries=2)
    assert len(llm.calls) == 2, "应当真的重试了 2 次"
    assert "不符合要求" in llm.calls[1]["user"], "第二次要带上上一次的错误"


# ── 解析 ──────────────────────────────────────────────────

def test_parse_pdf_extracts_pages_and_sections(env, pdf):
    ex, store, _ = env
    rep = ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1, targets=["parse"])
    paper = store.get(rep.results["parse"].artifact)
    assert len(paper["pages"]) == 3
    assert paper["char_count"] > 200
    titles = [s["title"] for s in paper["sections"]]
    assert any("Datasets" in t for t in titles), titles
    assert any(s["related_work"] for s in paper["sections"]), "应识别出相关工作章节"


def test_related_work_is_not_sent_to_model(env, pdf):
    """相关工作章节永远不进提示词 —— 那是误抽的主要来源。"""
    ex, _, llm = env
    ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1, targets=["extract"])
    prompt = llm.calls[0]["user"]
    assert "Datasets" in prompt
    assert "Related Work" not in prompt
    assert "rainfall" not in prompt


# ── 抽取与建议 ────────────────────────────────────────────

def test_extraction_flags_related_work_variable(env, pdf):
    ex, store, _ = env
    rep = ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1, targets=["extract"])
    out = store.get(rep.results["extract"].artifact)
    by = {v["name"]: v for v in out["variables"]}
    assert by["speed"]["suggest_adopt"] is True
    assert by["rainfall"]["suggest_adopt"] is False, "来自相关工作的变量默认不采纳"
    assert out["open_questions"], "论文没说清的地方要留给人"


def test_extraction_without_llm_fails_loudly(tmp_path, pdf):
    """没配模型时明确报错，不悄悄退化成规则抽取。"""
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")
    ex = Executor(store, rs, workdir=tmp_path, llm=None)
    rep = ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1, targets=["extract"])
    assert rep.results["extract"].status.value == "FAILED"
    assert "没有配置模型" in rep.results["extract"].error
    rs.close()


# ── ★ 人工确认闸门与续跑 ──────────────────────────────────

def test_pipeline_stops_at_confirmation(env, pdf):
    """没人确认过就必须停下，且中间产物全部保留。"""
    ex, _, _ = env
    rep = ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1)
    assert rep.status == "needs_human"
    assert rep.results["parse"].status.value == "COMPUTED"
    assert rep.results["extract"].status.value == "COMPUTED"
    assert "schema" not in rep.results or rep.results["schema"].status.value == "SKIPPED"

    ask = rep.trace.of_kind("ask")[0].data
    assert "speed" in ask["options"]
    assert set(ask["context"]["suggested"]) == {"speed", "volume", "occupancy"}


def test_resume_after_confirmation_reuses_cache(env, pdf):
    """★ 续跑是缓存的自然结果，没有单独的恢复逻辑。

    回填答案 → 只有 confirm 及其下游重算，parse/extract 全部命中缓存。
    对论文链路这一点很实在：extract 是要花钱的那一步。
    """
    ex, store, llm = env
    first = ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1)
    assert first.status == "needs_human"
    calls_after_first = len(llm.calls)

    decisions = {"adopt": ["speed", "volume", "occupancy"],
                 "answers": {"论文未给出原始缺失率，用多少？": "5%"}}
    second = ex.run(paper_pipeline(pdf_path=str(pdf), decisions=decisions), seed=1)

    assert second.ok, f"续跑应当跑通：{second}"
    assert second.results["parse"].status.value == "CACHED"
    assert second.results["extract"].status.value == "CACHED"
    assert second.results["confirm"].status.value == "COMPUTED"
    assert len(llm.calls) == calls_after_first, "续跑不该再调一次模型"

    schema = store.get(second.results["schema"].artifact)
    names = [f["name"] for f in schema["fields"]]
    assert names == ["speed", "volume", "occupancy"]
    assert "rainfall" not in names, "没采纳的变量不能混进 Schema"
    assert schema["target_field"] == "speed"


def test_rejecting_everything_fails_clearly(env, pdf):
    ex, _, _ = env
    ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1)
    rep = ex.run(paper_pipeline(pdf_path=str(pdf), decisions={"adopt": []}), seed=1)
    assert rep.results["confirm"].status.value == "FAILED"
    assert "一个变量都没采纳" in rep.results["confirm"].error


def test_human_overrides_are_applied_and_recorded(env, pdf):
    """人改了单位，要真的生效并记进血缘。"""
    ex, store, _ = env
    ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1)
    decisions = {
        "adopt": ["speed"],
        "overrides": {"speed": {"unit": "km/h", "max": 110}},
        "answers": {"speed 的单位？": "km/h（与 Table 2 一致）"},
    }
    rep = ex.run(paper_pipeline(pdf_path=str(pdf), decisions=decisions), seed=1)
    assert rep.ok

    confirmed = store.get(rep.results["confirm"].artifact)
    assert confirmed["variables"][0]["unit"] == "km/h"
    schema = store.get(rep.results["schema"].artifact)
    assert schema["fields"][0]["constraints"]["max"] == 110

    decisions_logged = rep.lineage.entries["confirm"].decisions
    assert any(d["answer"] == "km/h（与 Table 2 一致）" for d in decisions_logged)
    assert any("采纳哪些变量" in d["question"] for d in decisions_logged)


def test_changing_decisions_only_reruns_from_confirm(env, pdf):
    ex, _, llm = env
    ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1)
    ex.run(paper_pipeline(pdf_path=str(pdf), decisions={"adopt": ["speed", "volume"]}), seed=1)
    n = len(llm.calls)
    rep = ex.run(paper_pipeline(pdf_path=str(pdf), decisions={"adopt": ["speed"]}), seed=1)
    assert rep.results["extract"].status.value == "CACHED"
    assert rep.results["confirm"].status.value == "COMPUTED"
    assert len(llm.calls) == n


# ── 端到端 ────────────────────────────────────────────────

def test_paper_to_dataset_end_to_end(env, pdf):
    ex, store, _ = env
    ex.run(paper_pipeline(pdf_path=str(pdf)), seed=1)
    rep = ex.run(
        paper_pipeline(pdf_path=str(pdf), num_records=300,
                       decisions={"adopt": ["speed", "volume", "occupancy"]}),
        seed=1,
    )
    assert rep.ok, f"论文到数据集应当跑通：{rep}"

    df = store.get(rep.results["generate"].artifact)
    assert len(df) == 300
    assert list(df.columns) == ["speed", "volume", "occupancy"]
    assert df["speed"].between(0, 70).all()

    card = store.get(rep.results["export"].artifact)
    assert card["synthetic"] is True

    # 列级血缘要能追回论文的具体位置
    cols = rep.lineage.entries["confirm"].columns
    assert cols["speed"]["source_span"]["page"] == 3
    assert cols["speed"]["origin"] == "paper"
    assert "speed" in rep.lineage.explain("confirm")
