"""Planner：自然语言 → 计划书 → 流水线请求。

关键性质：**模型碰不到图的结构**。它只填一份扁平的计划书，
图由代码确定性地建出来。所以这里的测试重点在——
计划书里的各种"说不通"能不能被拦住或自动修好。
"""

from __future__ import annotations

import json

import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.llm.client import StubLLM
from dataforge.llm.contracts import ContractError, PLAN_SPEC, validate_against
from dataforge.planner import (
    Plan, PlanError, Planner, autofix, check_plan, plan_to_request,
)

GOOD = {
    "name": "traffic_flow",
    "pipeline": "tabular",
    "task_type": "classification",
    "domain": "traffic",
    "num_records": 800,
    "fields": [
        {"name": "road_id", "dtype": "string", "role": "structural"},
        {"name": "speed", "dtype": "float", "min": 0, "max": 70, "unit": "mph", "role": "input"},
        {"name": "volume", "dtype": "integer", "min": 0, "max": 1200, "role": "input"},
        {"name": "occupancy", "dtype": "float", "min": 0, "max": 1, "role": "input"},
        {"name": "congestion", "dtype": "categorical",
         "categories": ["free", "slow", "jam"], "role": "target"},
    ],
    "target": "congestion",
    "known_correlated": [["volume", "occupancy"]],
    "rationale": "交通拥堵分类，用速度/流量/占有率预测拥堵等级",
    "open_questions": [{"question": "样本量取多少？", "options": ["800", "5000"]}],
}


def spec(**over):
    s = json.loads(json.dumps(GOOD))
    s.update(over)
    return s


# ── 契约 ──────────────────────────────────────────────────

def test_good_plan_passes_contract_and_semantics():
    validate_against(GOOD, PLAN_SPEC)
    assert check_plan(GOOD) == []


def test_contract_rejects_unknown_pipeline():
    with pytest.raises(ContractError):
        validate_against(spec(pipeline="neural"), PLAN_SPEC)


def test_contract_rejects_zero_records():
    with pytest.raises(ContractError, match="不得小于"):
        validate_against(spec(num_records=0), PLAN_SPEC)


def test_contract_requires_at_least_one_field():
    with pytest.raises(ContractError, match="至少需要"):
        validate_against(spec(fields=[]), PLAN_SPEC)


# ── 语义校验 ──────────────────────────────────────────────

def test_rejects_target_not_in_fields():
    issues = check_plan(spec(target="nope"))
    assert any("不在字段列表" in i for i in issues)


def test_rejects_duplicate_field_names():
    s = spec()
    s["fields"].append({"name": "speed", "dtype": "float"})
    assert any("重复" in i for i in check_plan(s))


def test_rejects_bad_identifier():
    s = spec()
    s["fields"][0]["name"] = "道路编号"
    assert any("合法标识符" in i for i in check_plan(s))


def test_rejects_categorical_without_categories():
    s = spec()
    s["fields"][4].pop("categories")
    assert any("categories" in i for i in check_plan(s))


def test_rejects_timeseries_without_time_column():
    assert any("time_column" in i for i in check_plan(spec(pipeline="timeseries")))


def test_rejects_time_column_of_wrong_type():
    s = spec(pipeline="timeseries", time_column="speed")
    assert any("应为 datetime" in i for i in check_plan(s))


def test_rejects_forecast_task_on_tabular_pipeline():
    """时序预测走表格流水线会导致随机切分，虚高精度极具迷惑性 —— 必须拦。"""
    s = spec(task_type="time_series_forecast")
    s["fields"] = [f for f in s["fields"] if f["dtype"] != "datetime"]
    assert any("pipeline 不是 timeseries" in i for i in check_plan(s))


def test_rejects_known_correlated_with_unknown_field():
    assert any("known_correlated" in i for i in check_plan(spec(known_correlated=[["speed", "rain"]])))


# ── 自动修复 ──────────────────────────────────────────────

def test_autofix_swaps_reversed_min_max():
    s = spec()
    s["fields"][1]["min"], s["fields"][1]["max"] = 70, 0
    fixed, fixes = autofix(s)
    assert fixed["fields"][1]["min"] == 0 and fixed["fields"][1]["max"] == 70
    assert any("写反" in f for f in fixes)


def test_autofix_switches_to_timeseries_when_datetime_present():
    s = spec(task_type="time_series_forecast")
    s["fields"].insert(0, {"name": "ts", "dtype": "datetime",
                           "start": "2026-01-01", "end": "2026-03-01"})
    fixed, fixes = autofix(s)
    assert fixed["pipeline"] == "timeseries" and fixed["time_column"] == "ts"
    assert check_plan(fixed) == []
    assert any("timeseries" in f for f in fixes)


def test_autofix_corrects_target_by_role():
    fixed, fixes = autofix(spec(target="wrong_name"))
    assert fixed["target"] == "congestion"
    assert any("纠正" in f for f in fixes)


def test_autofix_clears_ambiguous_target():
    s = spec(target="wrong")
    for f in s["fields"]:
        f["role"] = "target"          # 多个候选 → 无法确定
    fixed, fixes = autofix(s)
    assert "target" not in fixed
    assert any("已清空" in f for f in fixes)


def test_autofix_drops_dangling_correlated_pairs():
    fixed, _ = autofix(spec(known_correlated=[["speed", "volume"], ["speed", "ghost"]]))
    assert fixed["known_correlated"] == [["speed", "volume"]]


def test_autofix_only_touches_unambiguous_problems():
    """有歧义的不自动改，留给 check_plan 报出来交给人。"""
    s = spec()
    s["fields"][0]["name"] = "道路编号"
    fixed, _ = autofix(s)
    assert fixed["fields"][0]["name"] == "道路编号"
    assert check_plan(fixed)


# ── 计划书 → 请求体 ───────────────────────────────────────

def test_plan_to_request_maps_fields():
    req = plan_to_request(GOOD, seed=7)
    assert req["kind"] == "tabular" and req["seed"] == 7
    assert req["target"] == "congestion"
    names = [f["name"] for f in req["fields"]]
    assert names == ["road_id", "speed", "volume", "occupancy", "congestion"]
    speed = next(f for f in req["fields"] if f["name"] == "speed")
    assert speed["type"] == "float" and speed["min"] == 0 and speed["unit"] == "mph"
    cong = next(f for f in req["fields"] if f["name"] == "congestion")
    assert cong["categories"] == ["free", "slow", "jam"]
    assert req["known_correlated"] == [["volume", "occupancy"]]


def test_plan_to_request_refuses_bad_plan():
    with pytest.raises(PlanError, match="不在字段列表"):
        plan_to_request(spec(target="ghost"))


def test_plan_to_request_carries_time_column():
    s = spec(pipeline="timeseries", task_type="time_series_forecast", time_column="ts")
    s["fields"].insert(0, {"name": "ts", "dtype": "datetime",
                           "start": "2026-01-01", "end": "2026-02-01"})
    req = plan_to_request(s)
    assert req["kind"] == "timeseries" and req["time_column"] == "ts"


# ── Planner ───────────────────────────────────────────────

def test_planner_returns_plan_and_request():
    llm = StubLLM(default=GOOD)
    p, req = Planner(llm).plan_request("给我一份交通拥堵分类数据集")
    assert p.ok and req is not None
    assert req["name"] == "traffic_flow"
    assert p.spec["open_questions"], "没说清的地方要留给人"


def test_planner_autofixes_before_reporting():
    """能确定性修好的先修掉，别为此再问一次模型。"""
    llm = StubLLM(default=spec(target="ghost", known_correlated=[]))
    p, req = Planner(llm).plan_request("随便来点数据")
    # 只有 congestion 标了 role=target，所以能唯一纠正
    assert p.ok and req is not None
    assert req["target"] == "congestion"
    assert any("纠正" in f for f in p.fixes)


def test_planner_without_llm_fails_loudly():
    with pytest.raises(PlanError, match="没有配置模型"):
        Planner(None).plan("给我数据")


def test_planner_rejects_empty_text():
    with pytest.raises(PlanError, match="空"):
        Planner(StubLLM(default=GOOD)).plan("   ")


def test_planner_retries_on_contract_violation():
    """契约不过要把错误喂回去重试，而不是把脏计划传下去。"""
    llm = StubLLM(default={"name": "x", "pipeline": "tabular"})   # 缺 fields
    with pytest.raises(Exception):
        Planner(llm).plan("来点数据")
    assert len(llm.calls) >= 2


def test_planner_caches_identical_text(tmp_path):
    """同一段需求重复规划不该重复花钱。"""
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")
    llm = StubLLM(default=GOOD)
    pl = Planner(llm, runstore=rs, store=store)

    first = pl.plan("交通拥堵分类")
    assert not first.cached and len(llm.calls) == 1

    second = pl.plan("交通拥堵分类")
    assert second.cached and len(llm.calls) == 1, "第二次不该再调模型"
    assert second.spec == first.spec
    rs.close()


def test_planner_cache_misses_on_different_text(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")
    llm = StubLLM(default=GOOD)
    pl = Planner(llm, runstore=rs, store=store)
    pl.plan("交通拥堵分类")
    pl.plan("医疗诊断数据")
    assert len(llm.calls) == 2
    rs.close()
