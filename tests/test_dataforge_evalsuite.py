"""测评集的难度分层、区分度与 Inspect 导出。

这个文件回答的是**「这套题本身好不好」**，而不是「被测 Agent 好不好」。
一套没有区分度的测评集，通过率再高也什么都没说明。
"""

from __future__ import annotations

import ast
import json

import pytest

from dataforge.graph.node import NodeContext
from dataforge.nodes.agenteval import check_task, generate_agent_eval, run_assertions
from dataforge.nodes.eval_sim import PROFILES, simulate, sweep
from dataforge.nodes.eval_templates import DIFFICULTIES, TEMPLATES, derive
from dataforge.nodes.inspect_export import export, task_to_sample


def ctx(seed=1):
    return NodeContext(node_id="t", run_id="t", seed=seed)


@pytest.fixture()
def tasks():
    return generate_agent_eval({}, {"num_tasks": 40}, ctx())["tasks"]


# ── 模板与分层 ────────────────────────────────────────────

def test_every_template_declares_difficulty():
    for t in TEMPLATES:
        assert t["difficulty"] in DIFFICULTIES, t["category"]


def test_all_difficulties_are_represented():
    got = {t["difficulty"] for t in TEMPLATES}
    assert got == set(DIFFICULTIES), f"缺少难度档：{set(DIFFICULTIES) - got}"


def test_generation_covers_all_difficulties(tasks):
    """★ 轮转取模板而不是随机抽——随机抽在小样本下会整档漏掉。"""
    small = generate_agent_eval({}, {"num_tasks": 10}, ctx())["tasks"]
    assert {t["difficulty"] for t in small} == set(DIFFICULTIES)


def test_difficulty_filter_works():
    only_hard = generate_agent_eval({}, {"num_tasks": 8, "difficulties": ["hard"]}, ctx())
    assert {t["difficulty"] for t in only_hard["tasks"]} == {"hard"}


def test_unknown_difficulty_rejected():
    with pytest.raises(ValueError, match="未知难度"):
        generate_agent_eval({}, {"difficulties": ["impossible"]}, ctx())


def test_empty_selection_gives_actionable_error():
    with pytest.raises(ValueError, match="可选难度"):
        generate_agent_eval({}, {"categories": ["safety"], "difficulties": ["medium"]}, ctx())


def test_all_generated_tasks_are_valid(tasks):
    bad = {t["task_id"]: check_task(t) for t in tasks if check_task(t)}
    assert not bad, bad


def test_derived_answers_match_prompt():
    """★ 答案必须与题面同源，否则整个测评集失去意义。"""
    for t in generate_agent_eval({}, {"num_tasks": 30}, ctx())["tasks"]:
        if t["category"] != "compute" or t["difficulty"] != "medium":
            continue
        nums = [int(x) for x in t["setup"]["files"]["nums.txt"].split()]
        k = int(t["prompt"].split("乘以 ")[1].split("，")[0])
        expected = next(a["value"] for a in t["assertions"] if a["kind"] == "value_equals")
        assert expected == sum(nums) * k, t["task_id"]


# ── 模拟 Agent ────────────────────────────────────────────

def test_noop_passes_nothing(tasks, tmp_path):
    """★ 什么都不做就一道都过不了 —— 否则判据是摆设。"""
    for i, t in enumerate(tasks):
        res = simulate(t, "noop", tmp_path / f"n{i}")
        assert not run_assertions(t, res, workdir=tmp_path / f"n{i}")["passed"], t["task_id"]


def test_perfect_passes_everything(tasks, tmp_path):
    """★ 正确做完就该全过 —— 否则是题目本身写错了。"""
    failed = []
    for i, t in enumerate(tasks):
        wd = tmp_path / f"p{i}"
        res = simulate(t, "perfect", wd)
        out = run_assertions(t, res, workdir=wd)
        if not out["passed"]:
            failed.append((t["task_id"], [c["message"] for c in out["failed"]]))
    assert not failed, failed


def test_unknown_profile_rejected(tasks, tmp_path):
    with pytest.raises(ValueError, match="未知档位"):
        simulate(tasks[0], "genius", tmp_path)


# ── 区分度 ────────────────────────────────────────────────

def test_suite_discriminates(tasks, tmp_path):
    """★ 本文件最重要的一条：五个能力档位的通过率必须拉得开。"""
    rep = sweep(tasks, tmp_path)
    rates = {p: rep["profiles"][p]["pass_rate"] for p in PROFILES}
    assert rates["perfect"] == 1.0, rates
    assert rates["noop"] == 0.0, rates
    assert rep["spread"] >= 0.9, rates
    assert rep["discriminates"] is True


def test_ordering_is_sensible(tasks, tmp_path):
    """能力越强通过率越高 —— 不满足的话判据在乱打分。"""
    r = sweep(tasks, tmp_path)["profiles"]
    assert r["perfect"]["pass_rate"] > r["wrong"]["pass_rate"]
    assert r["perfect"]["pass_rate"] > r["sloppy"]["pass_rate"]
    assert r["lazy"]["pass_rate"] > r["noop"]["pass_rate"] or \
        r["lazy"]["pass_rate"] == r["noop"]["pass_rate"] == 0.0


def test_safety_is_orthogonal_to_correctness(tasks, tmp_path):
    """★ 安全类测的是「该忍住时忍没忍住」，不是「算得对不对」。

    所以算错的 Agent 在安全题上应当照样通过，
    而违反约束的 Agent 应当挂掉——这两个方向不该混在一起。
    """
    r = sweep(tasks, tmp_path)["profiles"]
    assert r["wrong"]["by_category"]["safety"] == 1.0, "算错不该影响安全判据"
    assert r["sloppy"]["by_category"]["safety"] == 0.0, "违规必须被安全判据抓住"


def test_constrained_tasks_check_correctness_too(tasks, tmp_path):
    """★ 只测约束不测正确性，等于放过「守规矩但把事做砸」的 Agent。

    这条是 sweep 发现的真实缺口：constrained_hard 原本只查
    「文件存在 + 只写了一次」，写一堆垃圾也能过。
    """
    r = sweep(tasks, tmp_path)["profiles"]
    assert r["wrong"]["by_category"]["constrained"] == 0.0, \
        "写错内容的 Agent 不该通过 constrained 类任务"


def test_sweep_reports_its_own_limitation(tasks, tmp_path):
    """模拟结果必须自带局限说明，别让人当成真模型的成绩。"""
    rep = sweep(tasks, tmp_path)
    assert "不是真实模型" in rep["note"]


# ── Inspect 导出 ──────────────────────────────────────────

def test_sample_shape_matches_inspect(tasks):
    s = task_to_sample(tasks[0])
    assert {"id", "input", "target", "files", "metadata"} <= set(s)
    assert s["metadata"]["assertions"]
    assert s["metadata"]["difficulty"] in DIFFICULTIES


def test_export_writes_runnable_files(tasks, tmp_path):
    r = export(tasks, tmp_path / "out", generated_at="2026-08-24")
    assert r["num_samples"] == len(tasks)

    lines = (tmp_path / "out" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(tasks)
    for ln in lines:
        rec = json.loads(ln)
        assert rec["metadata"]["assertions"]


def test_generated_task_py_is_valid_python(tasks, tmp_path):
    """★ 模板里有大量 {{ }} 转义，最容易生成出语法错误的文件。"""
    r = export(tasks, tmp_path / "out")
    src = (tmp_path / "out" / "eval_task.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "run_assertions" in src, "判对错必须仍走 DataForge 的断言"
    assert "model_graded" not in src, "不能退回 LLM-as-Judge"


def test_export_readme_states_synthetic_and_limits(tasks, tmp_path):
    export(tasks, tmp_path / "out")
    md = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "不得用于刷榜" in md
    assert "尚未用真实模型验证" in md, "难度分层未经真模型验证这件事必须写明"


def test_setup_files_become_inspect_files(tasks):
    with_files = [t for t in tasks if t.get("setup", {}).get("files")]
    assert with_files, "应当有带初始文件的任务"
    s = task_to_sample(with_files[0])
    assert s["files"] and all(isinstance(v, str) for v in s["files"].values())
