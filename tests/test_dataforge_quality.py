"""泄漏检测、门禁、预算、实现漂移。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.graph.dag import DAG
from dataforge.graph.executor import Executor, NodeBlocked
from dataforge.graph.node import NodeSpec
from dataforge.harness.budget import Budget, BudgetExceeded, PlateauDetector
from dataforge.harness.gates import DEFAULTS, GateConfig, GateReport, gate_completeness, gate_row_count
from dataforge.quality import leakage as lk


@pytest.fixture()
def rng():
    return np.random.default_rng(20260823)


# ── 目标泄漏 ──────────────────────────────────────────────

def test_detects_target_leakage(rng):
    """特征几乎等于标签 —— 必须报 error。"""
    y = rng.normal(50, 10, 400)
    df = pd.DataFrame({
        "noise": rng.normal(0, 1, 400),
        "sneaky": y * 1.0001 + rng.normal(0, 0.01, 400),  # 标签的马甲
        "target": y,
    })
    found = lk.check_target_leakage(df, "target")
    errs = [f for f in found if f.severity == "error"]
    assert errs and "sneaky" in errs[0].columns
    assert lk.has_error(found)


def test_clean_features_are_not_flagged(rng):
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 300),
        "b": rng.normal(0, 1, 300),
        "target": rng.normal(0, 1, 300),
    })
    assert not lk.has_error(lk.check_target_leakage(df, "target"))


def test_known_correlated_pair_is_info_not_warning(rng):
    """物理规律导致的高相关不是泄漏。

    交通流里 occupancy 与 volume 天然强相关，报成泄漏纯属噪音——
    白名单里的只记 info。
    """
    base = rng.normal(100, 20, 400)
    df = pd.DataFrame({
        "volume": base,
        "occupancy": base * 0.9 + rng.normal(0, 3, 400),
    })
    plain = lk.check_target_leakage(df, "occupancy")
    assert any(f.severity in ("warning", "error") for f in plain)

    white = lk.check_target_leakage(
        df, "occupancy", known_correlated=[("volume", "occupancy")]
    )
    assert not lk.has_error(white)
    assert all(f.severity == "info" for f in white if f.columns)


def test_categorical_label_alias_is_detected():
    """分类目标：某列每个取值唯一对应一个标签 = 标签的另一种编码。"""
    df = pd.DataFrame({
        "code": ["A"] * 50 + ["B"] * 50,
        "other": list(range(100)),
        "label": ["yes"] * 50 + ["no"] * 50,
    })
    found = lk.check_target_leakage(df, "label")
    assert any(f.severity == "error" and "code" in f.columns for f in found)


# ── 切分重叠 ──────────────────────────────────────────────

def test_detects_split_overlap():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    splits = {"train": df.iloc[:80], "test": df.iloc[70:]}  # 10 行重叠
    found = lk.check_split_overlap(splits)
    assert found and found[0].severity == "error"
    assert found[0].details["count"] == 10


def test_disjoint_splits_are_clean():
    df = pd.DataFrame({"a": range(100)})
    assert not lk.check_split_overlap({"train": df.iloc[:70], "test": df.iloc[70:]})


# ── 时间穿越 ──────────────────────────────────────────────

def test_detects_temporal_leakage_from_random_split(rng):
    """随机切分时序数据 —— 时间上必然交叉，判 error。"""
    ts = pd.date_range("2026-01-01", periods=300, freq="h")
    df = pd.DataFrame({"t": ts, "v": rng.normal(size=300)})
    idx = rng.permutation(300)
    splits = {
        "train": df.iloc[idx[:210]],
        "val": df.iloc[idx[210:240]],
        "test": df.iloc[idx[240:]],
    }
    found = lk.check_temporal_leakage(splits, "t")
    assert found and all(f.severity == "error" for f in found)


def test_chronological_split_passes():
    ts = pd.date_range("2026-01-01", periods=300, freq="h")
    df = pd.DataFrame({"t": ts, "v": range(300)})
    splits = {"train": df.iloc[:210], "val": df.iloc[210:240], "test": df.iloc[240:]}
    assert not lk.check_temporal_leakage(splits, "t")


# ── 门禁 ──────────────────────────────────────────────────

def test_gate_blocks_on_row_count():
    df = pd.DataFrame({"a": [1, 2]})
    rep = GateReport().add(gate_row_count(df, GateConfig()))
    assert rep.blocked
    with pytest.raises(NodeBlocked):
        rep.raise_if_needed()


def test_gate_passes_when_healthy(rng):
    df = pd.DataFrame({"a": rng.normal(size=200)})
    cfg = GateConfig()
    rep = GateReport(config=cfg).add(gate_row_count(df, cfg)).add(gate_completeness(df, cfg))
    assert rep.passed
    rep.raise_if_needed()  # 不该抛


def test_threshold_override_is_audited():
    """阈值可以调，但必须留痕（设计文档 §3.3）。"""
    cfg = GateConfig()
    cfg.set("min_quality_score", 50.0, reason="赶时间")
    audit = cfg.audit()
    assert audit["relaxed"] is True
    o = audit["overrides"]["min_quality_score"]
    assert o["default"] == DEFAULTS["min_quality_score"]
    assert o["value"] == 50.0 and o["relaxed"] is True and o["reason"] == "赶时间"


def test_tightening_threshold_is_not_marked_relaxed():
    cfg = GateConfig()
    cfg.set("min_quality_score", 95.0)
    assert cfg.audit()["relaxed"] is False


def test_setting_back_to_default_clears_override():
    cfg = GateConfig()
    cfg.set("min_rows", 10)
    cfg.set("min_rows", DEFAULTS["min_rows"])
    assert cfg.audit()["overrides"] == {}


def test_unknown_threshold_rejected():
    with pytest.raises(KeyError):
        GateConfig().set("nope", 1)


# ── 预算与停滞 ────────────────────────────────────────────

def test_budget_stops_on_tokens():
    b = Budget(max_tokens=100)
    b.spend_tokens(90)
    with pytest.raises(BudgetExceeded):
        b.spend_tokens(20)


def test_budget_stops_on_repeated_error():
    """反复撞同一堵墙没有意义。"""
    b = Budget(max_retries_per_error=2)
    b.record_error("schema_invalid")
    b.record_error("schema_invalid")
    with pytest.raises(BudgetExceeded):
        b.record_error("schema_invalid")


def test_budget_counts_errors_separately():
    b = Budget(max_retries_per_error=2)
    for _ in range(2):
        b.record_error("a")
        b.record_error("b")  # 不同错误各自计数


def test_plateau_detects_no_progress():
    """★ 停滞检测：连续多轮没有实质进步就该停，别把预算烧满。"""
    p = PlateauDetector(epsilon=0.5, patience=2)
    assert p.push(60.0) is False
    assert p.push(72.0) is False
    assert p.push(72.1) is False          # 还没到 patience
    assert p.push(72.2) is True           # 连续两轮提升 < 0.5 → 停滞


def test_plateau_allows_real_progress():
    p = PlateauDetector(epsilon=0.5, patience=2)
    for s in (60.0, 70.0, 80.0, 90.0):
        assert p.push(s) is False
    assert p.best == 90.0


# ── 实现漂移 ──────────────────────────────────────────────

def test_drift_check_warns_when_impl_changes_without_version_bump(tmp_path):
    """改了实现却没升 code_version —— 缓存会喂回旧结果，必须报出来。

    这是设计文档 §5.2 点名的静默故障，我在写测试时自己踩过一次。
    """
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")

    def impl_v1(inputs, params, ctx):
        return [1, 2, 3]

    def impl_v2(inputs, params, ctx):
        return [9, 9, 9]          # 实现变了

    ex = Executor(store, rs, workdir=tmp_path)
    ex.run(DAG([NodeSpec("n", "GENERATE", impl_v1, code_version="1.0.0")]))

    with pytest.warns(RuntimeWarning, match="code_version"):
        rep = ex.run(DAG([NodeSpec("n", "GENERATE", impl_v2, code_version="1.0.0")]))
    assert rep.results["n"].status.value == "CACHED", "确实拿到了旧结果——所以才要报警"
    rs.close()


def test_strict_drift_blocks_the_run(tmp_path):
    """CI 里应当开 strict：这个故障静默的后果比停下来严重得多。"""
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")

    def impl_v1(inputs, params, ctx):
        return [1]

    def impl_v2(inputs, params, ctx):
        return [2]

    Executor(store, rs, workdir=tmp_path).run(
        DAG([NodeSpec("n", "GENERATE", impl_v1, code_version="1.0.0")])
    )
    strict = Executor(store, rs, workdir=tmp_path, strict_drift=True)
    rep = strict.run(DAG([NodeSpec("n", "GENERATE", impl_v2, code_version="1.0.0")]))
    assert rep.status == "blocked"
    assert "code_version" in rep.results["n"].error
    rs.close()


def test_no_drift_warning_when_version_bumped(tmp_path, recwarn):
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")

    def impl_v1(inputs, params, ctx):
        return [1]

    def impl_v2(inputs, params, ctx):
        return [2]

    ex = Executor(store, rs, workdir=tmp_path)
    ex.run(DAG([NodeSpec("n", "GENERATE", impl_v1, code_version="1.0.0")]))
    rep = ex.run(DAG([NodeSpec("n", "GENERATE", impl_v2, code_version="1.0.1")]))
    assert rep.results["n"].status.value == "COMPUTED"
    assert not [w for w in recwarn if "code_version" in str(w.message)]
    rs.close()
