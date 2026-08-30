"""M1 验收测试：图执行、指纹缓存、失效传播、重放。

设计文档 §11 定的两条验收标准就在这里：

1. ``test_replay_reconstructs_entire_run`` —— 任意一次运行可完整重放
2. ``test_change_one_param_only_reruns_affected_branch`` —— 改一个字段只重跑受影响分支
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.core.trace import Trace
from dataforge.graph.dag import DAG, CycleError
from dataforge.graph.executor import ExecStatus, Executor, NeedsHuman, NodeBlocked
from dataforge.graph.node import NodeSpec


# ── 测试用的可计数节点 ────────────────────────────────────

class Counter:
    """记录每个节点真正被执行了几次。缓存有没有生效，看这个。"""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def fn(self, name: str, produce):
        def _f(inputs, params, ctx):
            self.calls[name] = self.calls.get(name, 0) + 1
            return produce(inputs, params, ctx)
        return _f


@pytest.fixture()
def env(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    rs = RunStore(tmp_path / "runs.sqlite")
    ex = Executor(store, rs, workdir=tmp_path)
    yield store, rs, ex
    rs.close()


def build_dag(counter: Counter, *, n: int = 5, scale: int = 1) -> DAG:
    """一个有分叉的图::

        source ─┬─> left  ─┐
                │          ├─> merge
                └─> right ─┘
    """
    return DAG([
        NodeSpec("source", "SOURCE", counter.fn("source", lambda i, p, c: list(range(p["n"]))),
                 params={"n": n}, code_version="1.0.0"),
        NodeSpec("left", "TRANSFORM", counter.fn("left", lambda i, p, c: [x * p["scale"] for x in i["v"]]),
                 inputs={"v": "source"}, params={"scale": scale}, code_version="1.0.0"),
        NodeSpec("right", "TRANSFORM", counter.fn("right", lambda i, p, c: [x + 100 for x in i["v"]]),
                 inputs={"v": "source"}, code_version="1.0.0"),
        NodeSpec("merge", "EXPORT", counter.fn("merge", lambda i, p, c: {"l": i["a"], "r": i["b"]}),
                 inputs={"a": "left", "b": "right"}, code_version="1.0.0"),
    ])


# ── DAG 基础 ──────────────────────────────────────────────

def test_topo_order_is_deterministic():
    c = Counter()
    d = build_dag(c)
    assert d.topo_order() == d.topo_order()
    order = d.topo_order()
    assert order.index("source") < order.index("left") < order.index("merge")


def test_cycle_is_rejected():
    d = DAG([
        NodeSpec("a", "TRANSFORM", lambda i, p, c: 1, inputs={"x": "b"}, code_version="1"),
        NodeSpec("b", "TRANSFORM", lambda i, p, c: 1, inputs={"x": "a"}, code_version="1"),
    ])
    with pytest.raises(CycleError):
        d.validate()


def test_dangling_input_is_rejected():
    d = DAG([NodeSpec("a", "TRANSFORM", lambda i, p, c: 1, inputs={"x": "nope"}, code_version="1")])
    with pytest.raises(ValueError, match="不存在的节点"):
        d.validate()


def test_node_requires_code_version():
    with pytest.raises(ValueError, match="code_version"):
        NodeSpec("a", "SOURCE", lambda i, p, c: 1, code_version="")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="未知节点类型"):
        NodeSpec("a", "NOPE", lambda i, p, c: 1, code_version="1")


def test_descendants_and_ancestors():
    d = build_dag(Counter())
    assert d.descendants("source") == ["left", "merge", "right"]
    assert d.ancestors("merge") == ["left", "right", "source"]


# ── 执行与缓存 ────────────────────────────────────────────

def test_first_run_computes_everything(env):
    _, _, ex = env
    c = Counter()
    rep = ex.run(build_dag(c), seed=1)
    assert rep.ok
    assert sorted(rep.computed) == ["left", "merge", "right", "source"]
    assert rep.cached == []
    assert c.calls == {"source": 1, "left": 1, "right": 1, "merge": 1}


def test_second_identical_run_is_fully_cached(env):
    _, _, ex = env
    c = Counter()
    dag = build_dag(c)
    ex.run(dag, seed=1)
    rep2 = ex.run(dag, seed=1)
    assert sorted(rep2.cached) == ["left", "merge", "right", "source"]
    assert rep2.computed == []
    # 第二次一次都没真算
    assert c.calls == {"source": 1, "left": 1, "right": 1, "merge": 1}


def test_change_one_param_only_reruns_affected_branch(env):
    """★ M1 验收标准之二：改一个字段只重跑受影响的分支。

    改 left 的参数 —— source 和 right 不受影响应命中缓存，
    left 与其下游 merge 应重算。
    """
    _, _, ex = env
    c = Counter()
    ex.run(build_dag(c, scale=1), seed=1)
    rep = ex.run(build_dag(c, scale=2), seed=1)

    assert sorted(rep.computed) == ["left", "merge"], "只有 left 及其下游该重算"
    assert sorted(rep.cached) == ["right", "source"], "未受影响的分支该命中缓存"
    assert c.calls["source"] == 1, "source 不该被重算"
    assert c.calls["right"] == 1, "旁支 right 不该被重算"
    assert c.calls["left"] == 2 and c.calls["merge"] == 2


def test_code_version_bump_recomputes_that_node(env):
    """改了算法就必须重算该节点——否则会静默喂回旧数据（设计文档 §5.2）。"""
    _, _, ex = env
    c = Counter()
    ex.run(build_dag(c), seed=1)

    dag2 = build_dag(c)
    dag2["left"].code_version = "1.0.1"  # 假装改了实现
    rep = ex.run(dag2, seed=1)

    assert "left" in rep.computed, "升了 code_version 必须重算"
    assert {"right", "source"} <= set(rep.cached), "无关分支仍应命中"
    assert c.calls["left"] == 2


def test_invalidation_propagates_by_content_not_position(env):
    """失效按**内容**传播，不是按位置。

    这是内容寻址的直接推论，也是一条容易被误当成 bug 的性质：

    * 改了实现但输出逐字节相同 → 下游产物不会变，下游**不必**重算
    * 输出真的变了            → 下游哈希跟着变，下游必然重算

    第一种情况下强行重跑下游纯属浪费。
    """
    _, _, ex = env

    # ① 换实现但输出不变：下游应命中缓存
    c1 = Counter()
    ex.run(build_dag(c1), seed=1)
    same = build_dag(c1)
    same["left"].code_version = "9.9.9"
    rep_same = ex.run(same, seed=1)
    assert "left" in rep_same.computed
    assert "merge" in rep_same.cached, "上游内容没变，下游不该重算"

    # ② 输出真的变了：下游必须重算
    #   注意 code_version 必须与①不同。给两个不同实现用同一个版本号，
    #   会命中①的缓存拿到旧结果——这正是 §5.2 警告的静默故障，
    #   机制只能在遵守契约时保护你。
    c2 = Counter()
    ex.run(build_dag(c2), seed=1)
    changed = build_dag(c2)
    changed["left"].code_version = "9.9.10"
    changed["left"].fn = c2.fn("left", lambda i, p, c: [x * 1000 for x in i["v"]])
    rep_changed = ex.run(changed, seed=1)
    assert "left" in rep_changed.computed
    assert "merge" in rep_changed.computed, "上游内容变了，下游必须重算"


def test_seed_change_invalidates_all(env):
    _, _, ex = env
    c = Counter()
    dag = build_dag(c)
    ex.run(dag, seed=1)
    rep = ex.run(dag, seed=2)
    assert rep.cached == []


def test_force_reruns_node_and_descendants(env):
    _, _, ex = env
    c = Counter()
    dag = build_dag(c)
    ex.run(dag, seed=1)
    rep = ex.run(dag, seed=1, force=["left"])
    assert sorted(rep.computed) == ["left", "merge"]
    assert sorted(rep.cached) == ["right", "source"]


def test_uncacheable_node_always_recomputes(env):
    _, _, ex = env
    c = Counter()
    dag = build_dag(c)
    dag["source"].cacheable = False
    ex.run(dag, seed=1)
    rep = ex.run(dag, seed=1)
    assert "source" in rep.computed
    assert c.calls["source"] == 2


def test_targets_runs_minimal_subgraph(env):
    _, _, ex = env
    c = Counter()
    rep = ex.run(build_dag(c), seed=1, targets=["left"])
    assert set(rep.results) == {"source", "left"}
    assert "right" not in c.calls


# ── 失败与阻断 ────────────────────────────────────────────

def test_failure_skips_downstream_and_keeps_upstream(env):
    _, _, ex = env

    def boom(inputs, params, ctx):
        raise RuntimeError("炸了")

    dag = DAG([
        NodeSpec("a", "SOURCE", lambda i, p, c: [1, 2], code_version="1"),
        NodeSpec("b", "TRANSFORM", boom, inputs={"v": "a"}, code_version="1"),
        NodeSpec("c", "EXPORT", lambda i, p, c: i["v"], inputs={"v": "b"}, code_version="1"),
    ])
    rep = ex.run(dag)
    assert rep.status == "failed"
    assert rep.results["a"].status is ExecStatus.COMPUTED, "上游产物必须保留"
    assert rep.results["b"].status is ExecStatus.FAILED
    assert rep.results["c"].status is ExecStatus.SKIPPED
    assert "炸了" in rep.results["b"].error


def test_gate_block_stops_the_run(env):
    _, _, ex = env

    def gate(inputs, params, ctx):
        raise NodeBlocked("检出目标泄漏", gate="leakage")

    dag = DAG([
        NodeSpec("a", "SOURCE", lambda i, p, c: [1], code_version="1"),
        NodeSpec("g", "GATE", gate, inputs={"v": "a"}, code_version="1"),
        NodeSpec("e", "EXPORT", lambda i, p, c: 1, inputs={"v": "g"}, code_version="1"),
    ])
    rep = ex.run(dag)
    assert rep.status == "blocked"
    assert rep.results["g"].status is ExecStatus.BLOCKED
    assert rep.results["e"].status is ExecStatus.SKIPPED


def test_needs_human_pauses_and_keeps_artifacts(env):
    _, _, ex = env

    def ask(inputs, params, ctx):
        raise NeedsHuman("speed 的单位是 mph 还是 km/h？", options=["mph", "km/h"])

    dag = DAG([
        NodeSpec("a", "SOURCE", lambda i, p, c: [1], code_version="1"),
        NodeSpec("q", "GATE", ask, inputs={"v": "a"}, code_version="1"),
    ])
    rep = ex.run(dag)
    assert rep.status == "needs_human"
    assert rep.results["a"].status is ExecStatus.COMPUTED, "停下时中间产物不能丢"
    asks = rep.trace.of_kind("ask")
    assert asks and "mph" in asks[0].data["options"]


def test_failed_node_is_not_cached(env):
    """失败的节点不能进缓存，否则下次会「命中」一个不存在的产物。"""
    _, rs, ex = env
    calls = {"n": 0}

    def flaky(inputs, params, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("第一次失败")
        return [1, 2, 3]

    dag = DAG([NodeSpec("f", "GENERATE", flaky, code_version="1")])
    assert ex.run(dag).status == "failed"
    rep = ex.run(dag)
    assert rep.ok and rep.results["f"].status is ExecStatus.COMPUTED
    assert calls["n"] == 2


# ── 重放 ──────────────────────────────────────────────────

def test_replay_reconstructs_entire_run(env, tmp_path):
    """★ M1 验收标准之一：任意一次运行可完整重放。

    不重新执行，仅凭落盘的轨迹还原每个节点的指纹与产物哈希。
    """
    _, _, ex = env
    c = Counter()
    rep = ex.run(build_dag(c), seed=1)

    trace_file = tmp_path / "traces" / f"{rep.run_id}.jsonl"
    assert trace_file.exists()

    replayed = Trace.load(trace_file).replay()
    assert set(replayed) == set(rep.results)
    for nid, r in rep.results.items():
        assert replayed[nid]["fingerprint"] == r.fingerprint
        assert replayed[nid]["artifact"]["hash"] == r.artifact.hash
        assert replayed[nid]["status"] == r.status.value


def test_replay_distinguishes_cached_from_computed(env, tmp_path):
    _, _, ex = env
    c = Counter()
    dag = build_dag(c)
    ex.run(dag, seed=1)
    rep2 = ex.run(dag, seed=1)

    replayed = Trace.load(tmp_path / "traces" / f"{rep2.run_id}.jsonl").replay()
    assert all(v["status"] == "CACHED" for v in replayed.values())


def test_lineage_file_written(env, tmp_path):
    _, _, ex = env
    rep = ex.run(build_dag(Counter()), seed=1)
    assert (tmp_path / "lineage" / f"{rep.run_id}.json").exists()
    assert rep.lineage.ancestors("merge") == ["left", "right", "source"] or set(
        rep.lineage.ancestors("merge")) == {"left", "right", "source"}


def test_run_recorded_in_runstore(env):
    _, rs, ex = env
    rep = ex.run(build_dag(Counter()), seed=1)
    runs = rs.runs()
    assert runs[0]["run_id"] == rep.run_id
    assert runs[0]["status"] == "success"


def test_report_str_is_readable(env):
    _, _, ex = env
    rep = ex.run(build_dag(Counter()), seed=1)
    s = str(rep)
    assert rep.run_id in s and "source" in s and "COMPUTED" in s
