"""RL 轨迹 / LLM 训练集 / Agent 测评集。

三个生成器各有一条不能松的判据，这里的重点全在这三条上：

* RL —— **状态转移必须自洽**（重放动力学核对，统计指标看不出来）
* LLM —— **基准污染零容忍**（分数虚高看起来像"效果很好"）
* Agent 测评 —— **成功条件必须机器可判且有区分力**
"""

from __future__ import annotations

import json

import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.graph.executor import Executor
from dataforge.graph.node import NodeContext
from dataforge.nodes import agenteval as A
from dataforge.nodes import llmdata as L
from dataforge.nodes import rl as R
from dataforge.pipelines import agent_eval_pipeline, llm_sft_pipeline, rl_pipeline
from dataforge.quality.dedup import check_contamination, find_duplicates, normalize


@pytest.fixture()
def ex(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    rs = RunStore(tmp_path / "r.sqlite")
    yield Executor(store, rs, workdir=tmp_path), store
    rs.close()


def ctx(seed=1):
    return NodeContext(node_id="t", run_id="t", seed=seed)


# ══════════════ RL ══════════════

def test_gridworld_dynamics_are_deterministic():
    env = R.GridWorld(size=4, goal=(3, 3))
    s2, r, done = env.step({"row": 1, "col": 1}, "down")
    assert s2 == {"row": 2, "col": 1} and not done
    # 撞墙原地不动
    assert env.step({"row": 0, "col": 0}, "up")[0] == {"row": 0, "col": 0}
    # 到达目标给正奖励并终止
    s2, r, done = env.step({"row": 2, "col": 3}, "down")
    assert s2 == {"row": 3, "col": 3} and done and r == 1.0


def test_generated_trajectories_are_self_consistent():
    """★ 重放动力学核对每条转移。"""
    data = R.generate_trajectories({}, {"env": "gridworld", "num_episodes": 40}, ctx())
    findings = R.check_trajectories(data)
    assert not [f for f in findings if f["severity"] == "error"], findings


def test_inventory_trajectories_are_self_consistent():
    data = R.generate_trajectories(
        {}, {"env": "inventory", "num_episodes": 30, "max_steps": 20}, ctx())
    assert not [f for f in R.check_trajectories(data) if f["severity"] == "error"]
    # 需求作为转移的一部分被记下来，否则这份数据不可复核
    assert "demand" in data["episodes"][0]["steps"][0]


def test_checker_catches_tampered_next_state():
    """★ 这条测试证明检查器真的会红 —— 手工破坏一条转移。"""
    data = R.generate_trajectories({}, {"env": "gridworld", "num_episodes": 10}, ctx())
    data["episodes"][0]["steps"][0]["next_state"] = {"row": 99, "col": 99}
    errs = [f for f in R.check_trajectories(data) if f["severity"] == "error"]
    assert errs and any(f["kind"] in ("state_chain_broken", "dynamics_mismatch") for f in errs)


def test_checker_catches_tampered_reward():
    data = R.generate_trajectories({}, {"env": "gridworld", "num_episodes": 10}, ctx())
    data["episodes"][0]["steps"][0]["reward"] = 999.0
    errs = [f for f in R.check_trajectories(data) if f["severity"] == "error"]
    assert any(f["kind"] == "dynamics_mismatch" for f in errs)


def test_checker_catches_done_in_the_middle():
    data = R.generate_trajectories({}, {"env": "gridworld", "num_episodes": 10}, ctx())
    steps = data["episodes"][0]["steps"]
    if len(steps) < 2:
        pytest.skip("回合太短")
    steps[0]["done"] = True
    errs = [f for f in R.check_trajectories(data) if f["severity"] == "error"]
    assert any(f["kind"] == "done_not_last" for f in errs)


def test_trajectories_are_reproducible():
    a = R.generate_trajectories({}, {"env": "gridworld", "num_episodes": 20, "seed": 7}, ctx())
    b = R.generate_trajectories({}, {"env": "gridworld", "num_episodes": 20, "seed": 7}, ctx())
    assert a["episodes"] == b["episodes"]


def test_epsilon_controls_diversity():
    """全最优的轨迹里没有失败样本，离线 RL 学不出「什么不该做」。"""
    greedy = R.generate_trajectories(
        {}, {"env": "gridworld", "num_episodes": 40, "epsilon": 0.0}, ctx())
    noisy = R.generate_trajectories(
        {}, {"env": "gridworld", "num_episodes": 40, "epsilon": 0.9}, ctx())
    assert greedy["stats"]["mean_length"] < noisy["stats"]["mean_length"]


def test_bad_epsilon_rejected():
    with pytest.raises(ValueError, match="epsilon"):
        R.generate_trajectories({}, {"epsilon": 1.5}, ctx())


def test_unknown_env_rejected():
    with pytest.raises(ValueError, match="未知环境"):
        R.generate_trajectories({}, {"env": "atari"}, ctx())


def test_rl_pipeline_runs(ex):
    executor, store = ex
    rep = executor.run(rl_pipeline(num_episodes=50), seed=3)
    assert rep.ok, str(rep)
    data = store.get(rep.results["generate"].artifact)
    assert data["stats"]["num_episodes"] == 50
    assert store.get(rep.results["gate"].artifact)["passed"] is True


# ══════════════ 去重与污染 ══════════════

def test_normalize_ignores_punctuation_and_case():
    assert normalize("Hello, World!") == normalize("hello world")
    assert normalize("小明，有5个；苹果。") == normalize("小明 有5个 苹果")


def test_exact_duplicates_found():
    rep = find_duplicates(["一段完全相同的文字", "一段完全相同的文字", "另一段不同的话"])
    assert rep.unique == 2 and len(rep.exact) == 1


def test_near_duplicates_found():
    rep = find_duplicates([
        "把数据里的缺失值用中位数填充然后保存成csv文件",
        "把数据里的缺失值用中位数填充然后保存为csv文件",
        "完全不相干的另一个话题在讨论天气情况",
    ], threshold=0.75)
    assert rep.unique == 2, rep.to_dict()
    assert rep.near and rep.near[0].kind == "near"


def test_legitimate_variants_look_more_similar_than_real_duplicates():
    """★ 这条测试把一个原理性局限钉在测试里，防止有人"顺手"调低阈值。

    实测：合法变体（"中位数" vs "均值"）比真重复（只差一个字）**相似度还高**。
    字符级 Jaccard 分不开这两者，所以近重复只能报、不能自动删。
    """
    from dataforge.quality.dedup import jaccard, shingles

    real_dup = jaccard(
        shingles("把数据里的缺失值用中位数填充然后保存成csv文件"),
        shingles("把数据里的缺失值用中位数填充然后保存为csv文件"))
    legit_variant = jaccard(
        shingles("用 pandas 把 data.csv 里 age 列的缺失值填成中位数。"),
        shingles("用 pandas 把 data.csv 里 age 列的缺失值填成均值。"))
    assert legit_variant > real_dup, (legit_variant, real_dup)


def test_sft_only_removes_exact_duplicates():
    """近重复只标出、不删——自动删会悄悄抹掉合法的数据多样性。"""
    rows = [
        {"id": "a", "topic": "t", "instruction": "把缺失值填成中位数", "input": "", "output": "用 median()"},
        {"id": "b", "topic": "t", "instruction": "把缺失值填成中位数", "input": "", "output": "用 median()"},
        {"id": "c", "topic": "t", "instruction": "把缺失值填成均值", "input": "", "output": "用 mean()"},
    ]
    texts = [r["instruction"] + "\n" + r["output"] for r in rows]
    rep = find_duplicates(texts, threshold=0.75)
    exact = {p.j for p in rep.exact}
    assert exact == {1}, "只有完全相同的那条该被判为完全重复"
    assert len(rows) - len(exact) == 2, "均值那条是合法变体，必须保留"


def test_distinct_texts_are_not_merged():
    rep = find_duplicates(["今天下雨了要带伞", "明天开会记得准备材料", "股票市场今日大幅波动"])
    assert rep.unique == 3


def test_contamination_detected():
    """★ 训练集混进评测题目 —— 分数会虚高，看起来像效果很好。"""
    hits = check_contamination(
        ["小明有5个苹果又买了3个一共有多少个苹果", "一段完全无关的训练样本内容"],
        [("GSM8K", "小明有 5 个苹果，又买了 3 个，一共有多少个苹果？")],
    )
    assert len(hits) == 1 and hits[0].index == 0 and hits[0].benchmark == "GSM8K"


def test_contamination_threshold_is_looser_than_dedup():
    """污染是宁可错杀的场景，阈值刻意低于去重。"""
    import inspect
    assert inspect.signature(check_contamination).parameters["threshold"].default < \
        inspect.signature(find_duplicates).parameters["threshold"].default


def test_empty_reference_means_no_hits():
    assert check_contamination(["随便什么内容"], []) == []


# ══════════════ LLM 训练集 ══════════════

def test_sft_removes_exact_duplicates_before_returning():
    """完全重复自动删；近重复保留但必须统计出来让人看见。"""
    out = L.generate_sft({}, {"num_samples": 120}, ctx())
    assert out["format"] == "sft"
    assert out["stats"]["kept"] <= out["stats"]["generated"]

    texts = [s["instruction"] + "\n" + s["output"] for s in out["samples"]]
    assert find_duplicates(texts).exact == [], "产出里不该还有完全重复"
    assert "near_flagged" in out["stats"]
    assert out["stats"]["exact_removed"] + out["stats"]["kept"] == out["stats"]["generated"]


def test_dpo_pairs_have_checkable_advantage():
    """★ 偏好不能是随机指派的 —— 每组都要说得出 chosen 好在哪。"""
    sft = L.generate_sft({}, {"num_samples": 60}, ctx())
    dpo = L.generate_dpo({"sft": sft}, {}, ctx())
    assert dpo["pairs"]
    for p in dpo["pairs"]:
        assert p["chosen"].strip() != p["rejected"].strip()
        assert p["degrade_kind"] in L.DEGRADE
        assert p["why_chosen_better"]


def test_dpo_without_sft_fails_clearly():
    with pytest.raises(ValueError, match="没有可用的 SFT 样本"):
        L.generate_dpo({"sft": {"samples": []}}, {}, ctx())


def test_unknown_degrade_kind_rejected():
    sft = L.generate_sft({}, {"num_samples": 10}, ctx())
    with pytest.raises(ValueError, match="未知的劣化方式"):
        L.generate_dpo({"sft": sft}, {"degrade_kinds": ["make_it_worse"]}, ctx())


def test_llmdata_gate_blocks_on_contamination():
    """★ 污染零容忍。"""
    from dataforge.graph.executor import NodeBlocked

    polluted = {
        "format": "sft",
        "samples": [{"id": "x", "topic": "t", "instruction": "小明有5个苹果又买了3个",
                     "input": "", "output": "一共有多少个苹果？"}],
        "stats": {},
    }
    with pytest.raises(NodeBlocked, match="污染|contamination"):
        L.llmdata_gate({"dataset": polluted}, {}, ctx())


def test_llmdata_gate_passes_clean_data():
    sft = L.generate_sft({}, {"num_samples": 40}, ctx())
    out = L.llmdata_gate({"dataset": sft}, {}, ctx())
    assert out["passed"] is True
    assert any(r["gate"] == "contamination" and r["passed"] for r in out["results"])


def test_llm_pipeline_runs(ex):
    executor, store = ex
    rep = executor.run(llm_sft_pipeline(num_samples=60), seed=5)
    assert rep.ok, str(rep)
    dpo = store.get(rep.results["dpo"].artifact)
    assert dpo["stats"]["num_pairs"] > 0
    assert store.get(rep.results["dpo_gate"].artifact)["passed"] is True


# ══════════════ Agent 测评集 ══════════════

def test_assertion_kinds_exclude_arbitrary_code():
    """★ 测评集要发出去给别人跑，能塞任意代码就是远程执行漏洞。"""
    for bad in ("python", "eval", "exec", "shell", "command"):
        assert bad not in A.ASSERT_KINDS


def test_tool_called_assertion():
    r = {"output": "", "tool_calls": [{"name": "write_file"}], "error": None}
    assert A.check_assertion({"kind": "tool_called", "name": "write_file"}, r)["passed"]
    assert not A.check_assertion({"kind": "tool_called", "name": "read_file"}, r)["passed"]


def test_tool_not_called_assertion():
    """安全类任务的关键判据：不该做的事有没有做。"""
    clean = {"output": "", "tool_calls": [{"name": "read_file"}], "error": None}
    bad = {"output": "", "tool_calls": [{"name": "delete_file"}], "error": None}
    a = {"kind": "tool_not_called", "name": "delete_file"}
    assert A.check_assertion(a, clean)["passed"]
    assert not A.check_assertion(a, bad)["passed"]


def test_file_assertions(tmp_path):
    (tmp_path / "out.txt").write_text("hello bar world", encoding="utf-8")
    (tmp_path / "out.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    r = {"output": "", "tool_calls": [], "error": None}

    assert A.check_assertion({"kind": "file_exists", "path": "out.txt"}, r,
                             workdir=tmp_path)["passed"]
    assert not A.check_assertion({"kind": "file_exists", "path": "nope.txt"}, r,
                                 workdir=tmp_path)["passed"]
    assert A.check_assertion({"kind": "file_contains", "path": "out.txt", "value": "bar"}, r,
                             workdir=tmp_path)["passed"]
    assert A.check_assertion(
        {"kind": "file_json_path", "path": "out.json", "json_path": "status", "value": "ok"},
        r, workdir=tmp_path)["passed"]
    assert not A.check_assertion(
        {"kind": "file_json_path", "path": "out.json", "json_path": "status", "value": "bad"},
        r, workdir=tmp_path)["passed"]


def test_value_assertions():
    r = {"output": "结果是 20", "result": 20, "score": 0.7051, "tool_calls": [], "error": None}
    assert A.check_assertion({"kind": "value_equals", "path": "result", "value": 20}, r)["passed"]
    assert A.check_assertion({"kind": "value_close", "path": "score", "value": 0.705,
                              "tolerance": 0.01}, r)["passed"]
    assert not A.check_assertion({"kind": "value_close", "path": "score", "value": 0.9,
                                  "tolerance": 0.01}, r)["passed"]
    assert A.check_assertion({"kind": "value_in", "path": "result", "value": [10, 20]}, r)["passed"]
    assert A.check_assertion({"kind": "output_matches", "value": r"结果是 \d+"}, r)["passed"]


def test_missing_path_fails_rather_than_crashes():
    r = {"output": "", "tool_calls": [], "error": None}
    out = A.check_assertion({"kind": "value_equals", "path": "a.b.c", "value": 1}, r)
    assert out["passed"] is False and "缺失" in out["message"]


def test_bad_assertion_definition_is_caught():
    """一条跑不起来的断言比没有断言更糟 —— 它给人「有判据」的错觉。"""
    assert A.check_assertion_def({"kind": "nope"})
    assert A.check_assertion_def({"kind": "file_contains", "path": "x"})       # 缺 value
    assert A.check_assertion_def({"kind": "output_matches", "value": "["})     # 非法正则
    assert A.check_assertion_def({"kind": "value_in", "path": "x", "value": 1})  # 非列表
    assert not A.check_assertion_def({"kind": "tool_called", "name": "f"})


def test_task_without_assertions_is_rejected():
    """★ 每个样本必须有机器可判的成功条件。"""
    issues = A.check_task({"task_id": "t1", "prompt": "做点什么", "assertions": []})
    assert any("机器可判" in i for i in issues)


def test_assertion_referencing_missing_tool_is_caught():
    issues = A.check_task({
        "task_id": "t1", "prompt": "x",
        "tools": [{"name": "read_file"}],
        "assertions": [{"kind": "tool_called", "name": "write_file"}],
    })
    assert any("不在本任务的工具集里" in i for i in issues)


def test_generated_eval_set_is_valid():
    out = A.generate_agent_eval({}, {"num_tasks": 24}, ctx())
    assert out["stats"]["num_tasks"] == 24
    assert out["stats"]["assertions_per_task"] >= 1.5
    for t in out["tasks"]:
        assert A.check_task(t) == []


def test_run_assertions_requires_all_to_pass():
    """不做「通过一半算半分」—— 模糊化就退回 LLM-as-Judge 了。"""
    task = {"task_id": "t", "assertions": [
        {"kind": "tool_called", "name": "a"},
        {"kind": "tool_called", "name": "b"},
    ]}
    r = {"output": "", "tool_calls": [{"name": "a"}], "error": None}
    out = A.run_assertions(task, r)
    assert out["passed"] is False and len(out["failed"]) == 1


def test_eval_gate_rejects_trivially_passing_task():
    """★ 在「什么都不做」时也能通过的判据，区分不出好坏 Agent。"""
    from dataforge.graph.executor import NodeBlocked

    trivial = {
        "tasks": [{
            "task_id": "easy", "category": "x", "prompt": "随便",
            "tools": [{"name": "f"}],
            # 空跑时也成立
            "assertions": [{"kind": "tool_not_called", "name": "f"}, {"kind": "no_exception"}],
        }],
        "stats": {},
    }
    with pytest.raises(NodeBlocked, match="什么都不做"):
        A.agent_eval_gate({"eval_set": trivial}, {}, ctx())


def test_eval_gate_passes_real_tasks(tmp_path):
    out = A.generate_agent_eval({}, {"num_tasks": 12, "categories": ["compute"]}, ctx())
    gate = A.agent_eval_gate({"eval_set": out}, {"workdir": str(tmp_path)}, ctx())
    assert gate["passed"] is True


def test_agent_eval_pipeline_runs(ex):
    executor, store = ex
    rep = executor.run(agent_eval_pipeline(num_tasks=16, categories=["compute", "safety"]), seed=2)
    assert rep.ok, str(rep)
    data = store.get(rep.results["generate"].artifact)
    assert data["stats"]["num_tasks"] == 16
    assert set(data["stats"]["by_category"]) <= {"compute", "safety"}
    assert "LLM-as-Judge" in data["notice"]


def test_compute_task_assertion_actually_discriminates():
    """端到端：对的答案通过，错的答案不通过。"""
    out = A.generate_agent_eval({}, {"num_tasks": 1, "categories": ["compute"]}, ctx())
    task = out["tasks"][0]
    expected = next(a["value"] for a in task["assertions"] if a["kind"] == "value_equals")

    good = {"output": "", "result": expected, "tool_calls": [{"name": "calculator"}], "error": None}
    bad = {"output": "", "result": expected + 1, "tool_calls": [], "error": None}
    assert A.run_assertions(task, good)["passed"] is True
    assert A.run_assertions(task, bad)["passed"] is False


def test_contamination_checks_prompt_separately():
    """★ 题面要单独比，不能只比"问题+答案"整段。

    基准题目通常只对应样本的 instruction。拿整段去比，相似度会被长答案
    稀释到接近 0（实测 1.000 → 0.000），一条 prompt 逐字照抄评测题的样本
    反而检不出来——而这正是最该抓的一种污染。
    """
    from dataforge.graph.executor import NodeBlocked

    q = "什么是过拟合？用一两句话说清楚。"
    long_answer = ("过拟合指的是模型在训练集上表现很好、在新数据上明显变差的现象。"
                   "实际用的时候要注意训练集指标好不代表可用。" * 2)

    # 先确认：整段比的话确实检不出来
    from dataforge.quality.dedup import jaccard, shingles
    assert jaccard(shingles(q + "\n" + long_answer), shingles(q)) < 0.3

    polluted = {
        "format": "sft",
        "samples": [{"id": "x", "topic": "t", "instruction": q,
                     "input": "", "output": long_answer}],
        "stats": {},
    }
    with pytest.raises(NodeBlocked, match="污染|contamination"):
        L.llmdata_gate({"dataset": polluted},
                       {"reference": [("MyBench", q)]}, ctx())
