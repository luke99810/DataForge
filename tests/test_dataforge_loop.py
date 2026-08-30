"""PEVR 外层循环：打分、调参、五个出口。

设计文档 §4.2 的重点全在**停止判据**上。所以这里每个出口都有一条能变红的测试，
特别是 ``test_stops_on_plateau``（不停滞检测就会把预算烧完还看不出没进步）
和 ``test_refiner_never_touches_thresholds``（能调阈值的循环一定"收敛"）。
"""

from __future__ import annotations

import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.graph.executor import Executor
from dataforge.harness.budget import Budget, PlateauDetector, StopReason
from dataforge.harness.loop import RefineLoop
from dataforge.harness.refine import (
    LLMRefiner, RuleRefiner, assert_no_threshold_drift, sanitize,
)
from dataforge.harness.score import score_gate_report
from dataforge.llm.client import StubLLM
from dataforge.pipelines import tabular_pipeline

FIELDS = [
    {"name": "speed", "type": "float", "min": 0, "max": 70},
    {"name": "volume", "type": "integer", "min": 0, "max": 1200},
    {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
]


def gate_doc(*results, audit=None):
    return {
        "passed": all(r.get("passed") for r in results),
        "results": list(results),
        "audit": audit or {"thresholds": {}, "overrides": {}, "relaxed": False},
    }


def g(name, passed, level="WARN", message="", **details):
    return {"gate": name, "level": level, "passed": passed,
            "message": message, "details": details}


# ── 打分 ──────────────────────────────────────────────────

def test_perfect_run_scores_100():
    doc = gate_doc(g("row_count", True), g("leakage", True), g("completeness", True))
    assert score_gate_report(doc).score == 100.0


def test_missing_gate_report_scores_zero():
    b = score_gate_report(None)
    assert b.score == 0.0 and b.deductions


def test_block_costs_more_than_warn():
    blocked = score_gate_report(gate_doc(g("row_count", False, "BLOCK"))).score
    warned = score_gate_report(gate_doc(g("completeness", False, "WARN"))).score
    assert blocked < warned < 100.0


def test_leakage_error_is_heavily_penalised():
    doc = gate_doc(g("leakage", False, "BLOCK", findings=[
        {"severity": "error", "kind": "target"},
    ]))
    assert score_gate_report(doc).score <= 40.0


def test_completeness_penalty_is_continuous():
    """连续可导，否则「在变好但还不够」和「彻底卡住」看起来一样。"""
    def s(actual):
        return score_gate_report(
            gate_doc(g("completeness", False, "WARN", completeness=actual, min=0.9))
        ).score
    a, b, c = s(0.70), s(0.80), s(0.88)
    assert a < b < c, (a, b, c)


def test_row_count_penalty_rewards_getting_closer():
    def s(rows):
        return score_gate_report(
            gate_doc(g("row_count", False, "BLOCK", rows=rows, min=50))
        ).score
    assert s(5) < s(30) < s(49)


def test_breakdown_explains_itself():
    doc = gate_doc(g("row_count", False, "BLOCK", message="行数不够", rows=5, min=50))
    b = score_gate_report(doc)
    assert b.deductions and any("row_count" in d.get("gate", "") for d in b.deductions)
    assert b.signals["rows"] == 5


# ── 调参规则 ──────────────────────────────────────────────

def test_rule_refiner_raises_row_count():
    doc = gate_doc(g("row_count", False, "BLOCK", rows=5, min=50))
    p = RuleRefiner().propose({"num_records": 5}, breakdown={}, gate=doc)
    assert not p.give_up
    assert p.params["num_records"] >= 50
    assert p.changes[0]["param"] == "num_records"


def test_rule_refiner_lowers_missing_rate():
    doc = gate_doc(g("completeness", False, "WARN", completeness=0.5, min=0.9))
    p = RuleRefiner().propose({"missing_rate": 0.5}, breakdown={}, gate=doc)
    assert not p.give_up and p.params["missing_rate"] < 0.5


def test_rule_refiner_gives_up_on_leakage():
    """泄漏调参救不了 —— 必须明说，而不是继续试到预算耗尽。"""
    doc = gate_doc(g("leakage", False, "BLOCK", message="检出目标泄漏"))
    p = RuleRefiner().propose({"num_records": 100}, breakdown={}, gate=doc)
    assert p.give_up and "泄漏" in p.give_up_reason


def test_rule_refiner_gives_up_when_no_lever():
    doc = gate_doc(g("completeness", False, "WARN", completeness=0.5, min=0.9))
    p = RuleRefiner().propose({"missing_rate": 0.0}, breakdown={}, gate=doc)
    assert p.give_up and "没有着力点" in p.give_up_reason


def test_rule_refiner_gives_up_on_unknown_failure():
    doc = gate_doc(g("mystery", False, "WARN", message="说不清"))
    p = RuleRefiner().propose({"num_records": 100}, breakdown={}, gate=doc)
    assert p.give_up and "没有可调的参数" in p.give_up_reason


# ── 阈值红线 ──────────────────────────────────────────────

def test_sanitize_drops_non_tunable_params():
    """模型返回什么都行，能落地的只有白名单。"""
    clean, changes = sanitize(
        {"num_records": 100, "thresholds": {"min_rows": 50}},
        {"num_records": 200, "thresholds": {"min_rows": 1}, "fields": ["恶意"]},
    )
    assert clean["num_records"] == 200
    assert clean["thresholds"] == {"min_rows": 50}, "阈值不能被改"
    assert [c["param"] for c in changes] == ["num_records"]


def test_sanitize_rejects_out_of_range():
    clean, changes = sanitize({"missing_rate": 0.1}, {"missing_rate": 5.0})
    assert clean["missing_rate"] == 0.1 and not changes


def test_threshold_drift_assertion_fires():
    with pytest.raises(AssertionError, match="阈值"):
        assert_no_threshold_drift({"thresholds": {"min_rows": 50}},
                                  {"thresholds": {"min_rows": 1}})


def test_rule_refiner_never_emits_thresholds():
    doc = gate_doc(g("row_count", False, "BLOCK", rows=5, min=50))
    p = RuleRefiner().propose(
        {"num_records": 5, "thresholds": {"min_rows": 50}}, breakdown={}, gate=doc
    )
    assert p.params["thresholds"] == {"min_rows": 50}


# ── LLM 兜底 ──────────────────────────────────────────────

def test_llm_refiner_not_called_when_rules_suffice():
    """规则能处理就不花钱问模型。"""
    llm = StubLLM(default={"params": {}, "rationale": "x"})
    doc = gate_doc(g("row_count", False, "BLOCK", rows=5, min=50))
    LLMRefiner(llm).propose({"num_records": 5}, breakdown={}, gate=doc)
    assert llm.calls == []


def test_llm_refiner_kicks_in_when_rules_give_up():
    llm = StubLLM(default={"params": {"num_records": 999}, "rationale": "试试更多样本"})
    doc = gate_doc(g("mystery", False, "WARN"))
    p = LLMRefiner(llm).propose({"num_records": 100}, breakdown={"score": 40}, gate=doc)
    assert llm.calls and not p.give_up
    assert p.params["num_records"] == 999


def test_llm_proposal_touching_thresholds_is_discarded():
    llm = StubLLM(default={"params": {"thresholds": {"min_rows": 1}}, "rationale": "降低标准"})
    doc = gate_doc(g("mystery", False, "WARN"))
    p = LLMRefiner(llm).propose(
        {"num_records": 100, "thresholds": {"min_rows": 50}}, breakdown={}, gate=doc
    )
    assert p.give_up, "只想改阈值的提议必须被丢弃"
    assert "越界" in p.give_up_reason


def test_llm_failure_falls_back_to_rule_conclusion():
    llm = StubLLM(default="这不是 JSON")
    doc = gate_doc(g("mystery", False, "WARN"))
    p = LLMRefiner(llm).propose({"num_records": 100}, breakdown={}, gate=doc)
    assert p.give_up and "模型兜底也失败" in p.give_up_reason


# ── 循环出口 ──────────────────────────────────────────────

@pytest.fixture()
def ex(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")
    yield Executor(store, rs, workdir=tmp_path)
    rs.close()


def build(params):
    return tabular_pipeline(
        fields=FIELDS, name="loop", target="level",
        num_records=params["num_records"],
        missing_rate=params.get("missing_rate", 0.0),
        thresholds=params.get("thresholds", {}),
    )


def test_loop_succeeds_first_round(ex):
    loop = RefineLoop(ex, build)
    rep = loop.run({"num_records": 300, "thresholds": {}}, seed=1)
    assert rep.reason is StopReason.SUCCESS
    assert len(rep.rounds) == 1 and rep.rounds[0].score == 100.0


def test_loop_refines_row_count_then_succeeds(ex):
    """★ 第一轮被门禁拦下 → 自动加行数 → 第二轮通过。"""
    loop = RefineLoop(ex, build)
    rep = loop.run({"num_records": 5, "thresholds": {}}, seed=1)
    assert rep.reason is StopReason.SUCCESS, str(rep)
    assert len(rep.rounds) >= 2
    assert rep.rounds[0].status == "blocked"
    assert rep.rounds[0].proposal["changes"][0]["param"] == "num_records"
    assert rep.rounds[-1].score == 100.0
    assert rep.scores[-1] > rep.scores[0], "分数要真的涨了"


def test_loop_stops_blocked_when_unfixable(ex):
    """调参救不了的失败要立刻停，不能把预算烧完。"""
    class Leaky(RuleRefiner):
        def propose(self, params, *, breakdown, gate, round_no=1):
            p = super().propose(params, breakdown=breakdown, gate=gate, round_no=round_no)
            p.give_up = True
            p.give_up_reason = "检出目标泄漏，改参数修不好"
            return p

    loop = RefineLoop(ex, build, refiner=Leaky(), budget=Budget(max_rounds=9))
    rep = loop.run({"num_records": 5, "thresholds": {}}, seed=1)
    assert rep.reason is StopReason.BLOCKED
    assert len(rep.rounds) == 1, "该停就停，不该继续试"


def test_stops_on_plateau(ex):
    """★ 停滞检测：没有它，Agent 会把预算烧完却从第 2 轮起就在原地打转。"""
    class Noop(RuleRefiner):
        def propose(self, params, *, breakdown, gate, round_no=1):
            from dataforge.harness.refine import Proposal
            # 假装改了点什么，但实际不影响结果 —— 模拟"看起来在努力"
            return Proposal(params=dict(params),
                            changes=[{"param": "noise_ratio", "from": 0, "to": 0, "why": "试试"}],
                            rationale="继续试")

    loop = RefineLoop(
        ex, build, refiner=Noop(),
        plateau=PlateauDetector(epsilon=0.5, patience=2),
        budget=Budget(max_rounds=20),
    )
    rep = loop.run({"num_records": 5, "thresholds": {}}, seed=1)
    assert rep.reason is StopReason.PLATEAU
    assert len(rep.rounds) < 20, "停滞要在预算耗尽之前触发"
    assert "已达当前方法上限" in rep.message


def test_stops_when_budget_exhausted(ex):
    class Noop(RuleRefiner):
        def propose(self, params, *, breakdown, gate, round_no=1):
            from dataforge.harness.refine import Proposal
            return Proposal(params=dict(params),
                            changes=[{"param": "x", "from": 0, "to": 1, "why": "试试"}])

    loop = RefineLoop(
        ex, build, refiner=Noop(),
        plateau=PlateauDetector(epsilon=0.0, patience=999),   # 关掉停滞检测
        budget=Budget(max_rounds=3),
    )
    rep = loop.run({"num_records": 5, "thresholds": {}}, seed=1)
    assert rep.reason is StopReason.EXHAUSTED
    assert len(rep.rounds) == 3


def test_loop_reports_best_round(ex):
    loop = RefineLoop(ex, build)
    rep = loop.run({"num_records": 5, "thresholds": {}}, seed=1)
    assert rep.best is not None
    assert rep.best.score == max(rep.scores)


def test_best_marker_points_at_best_round(ex):
    """星号要标在真正的最佳轮上（index 从 1 起、best_index 是下标，差一位很容易）。"""
    rep = RefineLoop(ex, build).run({"num_records": 5, "thresholds": {}}, seed=1)
    assert len(rep.rounds) >= 2
    lines = str(rep).splitlines()[1:]
    marked = [i for i, ln in enumerate(lines) if ln.startswith(" *")]
    assert marked == [rep.best_index], f"星号在第 {marked} 行，最佳轮下标是 {rep.best_index}"


def test_loop_str_is_readable(ex):
    rep = RefineLoop(ex, build).run({"num_records": 5, "thresholds": {}}, seed=1)
    s = str(rep)
    assert "第1轮" in s and "分数" in s


def test_loop_reuses_cache_across_rounds(ex):
    """轮与轮之间共享缓存：schema 不随行数变，不该反复重算。"""
    loop = RefineLoop(ex, build)
    rep = loop.run({"num_records": 5, "thresholds": {}}, seed=1)
    assert len(rep.rounds) >= 2
    stats = ex.runstore.cache_stats()
    assert stats["total_hits"] > 0, "第二轮应当命中过缓存"
