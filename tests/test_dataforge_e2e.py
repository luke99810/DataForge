"""端到端：编排层与现有内核确实接得上。

不是「模块能 import」，是真的跑出一份数据、过了门禁、带上了数据卡。
"""

from __future__ import annotations

import pandas as pd
import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.runstore import RunStore
from dataforge.core.trace import Trace
from dataforge.graph.executor import Executor
from dataforge.pipelines import tabular_pipeline, timeseries_pipeline

FIELDS = [
    {"name": "sensor_id", "type": "string", "description": "传感器编号"},
    {"name": "speed", "type": "float", "min": 0, "max": 70, "unit": "mph",
     "origin": "paper", "source_span": {"page": 7, "section": "4.1"}},
    {"name": "volume", "type": "integer", "min": 0, "max": 1200},
    {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
]


@pytest.fixture()
def ex(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    rs = RunStore(tmp_path / "runs.sqlite")
    yield Executor(store, rs, workdir=tmp_path), store, tmp_path
    rs.close()


def test_tabular_pipeline_end_to_end(ex):
    executor, store, tmp = ex
    dag = tabular_pipeline(fields=FIELDS, num_records=300, name="traffic", target="level")
    rep = executor.run(dag, seed=20260823, label="e2e")

    assert rep.ok, f"流水线未通过：{rep}"
    assert set(rep.results) == {"schema", "generate", "analyze", "gate", "export"}

    df = store.get(rep.results["generate"].artifact)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 300
    assert {"sensor_id", "speed", "volume", "level"} <= set(df.columns)
    assert df["speed"].between(0, 70).all(), "生成值必须落在 Schema 约束内"

    gate = store.get(rep.results["gate"].artifact)
    assert gate["passed"] is True
    assert any(r["gate"] == "leakage" for r in gate["results"])

    card = store.get(rep.results["export"].artifact)
    assert card["synthetic"] is True, "合成标记不可缺失（设计文档 §12 红线一）"
    assert "不得用于刷榜" in card["notice"]
    assert card["rows"] == 300


def test_pipeline_is_reproducible(ex):
    """同 seed 同参数 → 同产物哈希。做不到就谈不上可复现。"""
    executor, _, _ = ex
    dag = tabular_pipeline(fields=FIELDS, num_records=200, name="t", target="level")
    r1 = executor.run(dag, seed=7)
    executor.runstore.cache_clear()          # 清缓存，逼它真算第二遍
    r2 = executor.run(dag, seed=7)

    assert r2.computed, "缓存已清，应当真的重算"
    assert r1.results["generate"].artifact.hash == r2.results["generate"].artifact.hash


def test_changing_num_records_reuses_schema(ex):
    """★ 改一个参数只重跑受影响分支 —— 在真实流水线上再验一次。"""
    executor, _, _ = ex
    executor.run(tabular_pipeline(fields=FIELDS, num_records=200, name="t", target="level"), seed=7)
    rep = executor.run(tabular_pipeline(fields=FIELDS, num_records=400, name="t", target="level"), seed=7)

    assert "schema" in rep.cached, "schema 与 num_records 无关，不该重算"
    assert {"generate", "analyze", "gate", "export"} <= set(rep.computed)


def test_changing_a_field_invalidates_everything(ex):
    executor, _, _ = ex
    executor.run(tabular_pipeline(fields=FIELDS, num_records=200, name="t", target="level"), seed=7)

    changed = [dict(f) for f in FIELDS]
    changed[1]["max"] = 100          # speed 上限改了
    rep = executor.run(tabular_pipeline(fields=changed, num_records=200, name="t", target="level"), seed=7)
    assert "schema" in rep.computed and "generate" in rep.computed


def test_timeseries_pipeline_splits_chronologically(ex):
    executor, store, _ = ex
    fields = [
        {"name": "ts", "type": "datetime", "start": "2026-01-01", "end": "2026-03-01"},
        {"name": "speed", "type": "float", "min": 0, "max": 70},
    ]
    dag = timeseries_pipeline(fields=fields, time_column="ts", num_records=500, name="ts_demo")
    rep = executor.run(dag, seed=1)
    assert rep.ok, f"时序流水线未通过：{rep}"

    splits = store.get(rep.results["split"].artifact)
    assert set(splits) == {"train", "val", "test"}
    train, test = pd.DataFrame(splits["train"]), pd.DataFrame(splits["test"])
    assert len(train) > len(test) > 0
    # 时间上严格前后，不重叠
    assert pd.to_datetime(train["ts"]).max() <= pd.to_datetime(test["ts"]).min()


def test_gate_blocks_when_rows_too_few(ex):
    """门禁真的会拦住，而不是「有问题但先导出来看看」。"""
    executor, _, _ = ex
    dag = tabular_pipeline(fields=FIELDS, num_records=5, name="tiny", target="level")
    rep = executor.run(dag, seed=1)
    assert rep.status == "blocked"
    assert rep.results["gate"].status.value == "BLOCKED"
    assert rep.results["export"].status.value == "SKIPPED", "被阻断就不该导出"


def test_relaxed_threshold_is_recorded_in_data_card(ex):
    """放宽阈值让它过 —— 允许，但必须写进数据卡让下游看得见。"""
    executor, store, _ = ex
    dag = tabular_pipeline(
        fields=FIELDS, num_records=5, name="tiny", target="level",
        thresholds={"min_rows": 1},
    )
    rep = executor.run(dag, seed=1)
    assert rep.ok

    card = store.get(rep.results["export"].artifact)
    audit = card["gate_audit"]
    assert audit["relaxed"] is True
    assert audit["overrides"]["min_rows"]["value"] == 1
    assert audit["overrides"]["min_rows"]["default"] == 50


def test_lineage_carries_column_provenance(ex):
    """列级来源要能追回论文的具体位置。"""
    executor, _, tmp = ex
    rep = executor.run(
        tabular_pipeline(fields=FIELDS, num_records=100, name="t", target="level"), seed=3
    )
    entry = rep.lineage.entries["schema"]
    assert entry.columns["speed"]["source_span"] == {"page": 7, "section": "4.1"}
    assert entry.columns["speed"]["unit"] == "mph"
    assert "speed" in rep.lineage.explain("schema")


def test_full_run_replays_from_trace(ex):
    """★ 任意一次运行可完整重放 —— 在真实流水线上再验一次。"""
    executor, _, tmp = ex
    rep = executor.run(
        tabular_pipeline(fields=FIELDS, num_records=150, name="t", target="level"), seed=11
    )
    replayed = Trace.load(tmp / "traces" / f"{rep.run_id}.jsonl").replay()
    for nid, r in rep.results.items():
        assert replayed[nid]["fingerprint"] == r.fingerprint
        assert replayed[nid]["artifact"]["hash"] == r.artifact.hash
