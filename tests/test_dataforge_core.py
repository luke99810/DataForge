"""L0 地基测试：指纹、制品库、轨迹、血缘。"""

from __future__ import annotations

import pandas as pd
import pytest

from dataforge.core.artifacts import ArtifactStore
from dataforge.core.fingerprint import fingerprint, hash_value
from dataforge.core.lineage import Lineage
from dataforge.core.runstore import RunStore
from dataforge.core.trace import Trace


# ── 指纹 ──────────────────────────────────────────────────

def test_hash_is_content_addressed():
    a = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    b = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    c = pd.DataFrame({"x": [1, 2, 4], "y": ["a", "b", "c"]})
    assert hash_value(a) == hash_value(b)
    assert hash_value(a) != hash_value(c)


def test_hash_sees_column_order_and_dtype():
    a = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    assert hash_value(a) != hash_value(a[["y", "x"]])
    assert hash_value(a) != hash_value(a.astype("float64"))


def test_dict_hash_is_order_insensitive():
    assert hash_value({"a": 1, "b": 2}) == hash_value({"b": 2, "a": 1})


def test_code_version_is_mandatory():
    """缺 code_version 必须直接报错。

    这是设计文档 §5.2 点名的隐蔽 bug：不含代码版本时改了算法指纹不变，
    缓存命中返回旧数据且不报错。
    """
    with pytest.raises(ValueError, match="code_version"):
        fingerprint(node_id="n", kind="GENERATE", upstream={}, params={}, code_version="")


def test_code_version_change_invalidates_fingerprint():
    kw = dict(node_id="n", kind="GENERATE", upstream={"a": "h1"}, params={"k": 1})
    assert fingerprint(**kw, code_version="1.0.0") != fingerprint(**kw, code_version="1.0.1")


def test_upstream_change_invalidates_fingerprint():
    kw = dict(node_id="n", kind="GENERATE", params={}, code_version="1.0.0")
    assert fingerprint(**kw, upstream={"a": "h1"}) != fingerprint(**kw, upstream={"a": "h2"})


def test_fingerprint_is_stable_across_calls():
    kw = dict(
        node_id="n", kind="GENERATE", upstream={"a": "h1"},
        params={"z": 1, "a": [1, 2]}, code_version="1.0.0", seed=7,
    )
    assert fingerprint(**kw) == fingerprint(**kw)


# ── 制品库 ────────────────────────────────────────────────

def test_artifact_roundtrip_dataframe(tmp_path):
    store = ArtifactStore(tmp_path / "art")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5]})
    ref = store.put(df)
    assert ref.kind == "dataframe"
    assert ref.meta["rows"] == 3 and ref.meta["cols"] == 2
    pd.testing.assert_frame_equal(store.get(ref), df)


@pytest.mark.parametrize("value", [
    {"a": 1, "b": [1, 2]},
    ["x", "y"],
    "纯文本",
    b"\x00\x01\x02",
    42,
])
def test_artifact_roundtrip_types(tmp_path, value):
    store = ArtifactStore(tmp_path / "art")
    assert store.get(store.put(value)) == value


def test_artifact_dedup(tmp_path):
    """同样的内容只落一份盘。"""
    store = ArtifactStore(tmp_path / "art")
    df = pd.DataFrame({"a": [1, 2]})
    r1, r2 = store.put(df), store.put(df.copy())
    assert r1.hash == r2.hash
    assert store.path_of(r1) == store.path_of(r2)


def test_artifact_missing_raises(tmp_path):
    store = ArtifactStore(tmp_path / "art")
    ref = store.put({"a": 1})
    store.path_of(ref).unlink()
    with pytest.raises(FileNotFoundError):
        store.get(ref)


# ── 轨迹 ──────────────────────────────────────────────────

def test_trace_persists_and_reloads(tmp_path):
    p = tmp_path / "t.jsonl"
    with Trace("run_1", p) as t:
        t.emit("run_start", run_id="run_1")
        t.emit("node_end", "n1", fingerprint="fp1", artifact={"hash": "h1", "kind": "json"})
        t.emit("run_end", run_id="run_1", status="success")

    back = Trace.load(p)
    assert back.run_id == "run_1"
    assert len(back) == 3
    assert [e.kind for e in back] == ["run_start", "node_end", "run_end"]


def test_trace_replay_reconstructs_node_outputs(tmp_path):
    """M0 验收：仅凭轨迹就能还原每个节点算出了什么。"""
    p = tmp_path / "t.jsonl"
    with Trace("r", p) as t:
        t.emit("node_end", "a", fingerprint="fpa", artifact={"hash": "ha", "kind": "json"})
        t.emit("node_cached", "b", fingerprint="fpb", artifact={"hash": "hb", "kind": "json"})

    rep = Trace.load(p).replay()
    assert rep["a"] == {"fingerprint": "fpa", "artifact": {"hash": "ha", "kind": "json"},
                        "status": "COMPUTED"}
    assert rep["b"]["status"] == "CACHED"


def test_trace_seq_is_monotonic(tmp_path):
    with Trace("r", tmp_path / "t.jsonl") as t:
        for i in range(5):
            t.emit("note", note=i)
    assert [e.seq for e in t] == [0, 1, 2, 3, 4]


# ── 血缘 ──────────────────────────────────────────────────

def test_lineage_tracks_ancestors_and_decisions():
    lin = Lineage()
    lin.record("src", kind="SOURCE", fingerprint="f1", artifact="h1")
    lin.record("gen", kind="GENERATE", fingerprint="f2", artifact="h2",
               upstream={"schema": "src"}, params={"n": 10})
    lin.add_decision("gen", question="speed 的单位？", answer="mph",
                     note="论文正文与 Table 2 冲突")
    lin.add_column_source("gen", "speed", {"paper": "x.pdf", "page": 7})

    assert lin.ancestors("gen") == ["src"]
    assert lin.descendants("src") == ["gen"]
    text = lin.explain("gen")
    assert "mph" in text and "speed" in text and "src" in text


def test_lineage_saves_json(tmp_path):
    lin = Lineage()
    lin.record("a", kind="SOURCE", fingerprint="f", artifact="h")
    out = tmp_path / "l" / "lin.json"
    lin.save(out)
    assert out.exists() and '"a"' in out.read_text(encoding="utf-8")


# ── 运行记录 ──────────────────────────────────────────────

def test_runstore_cache_roundtrip(tmp_path):
    from dataforge.core.artifacts import ArtifactRef

    rs = RunStore(tmp_path / "db.sqlite")
    ref = ArtifactRef(hash="h1", kind="json", meta={"size": 3})
    assert rs.cache_get("fp1") is None
    rs.cache_put("fp1", "n1", "GENERATE", ref)
    got = rs.cache_get("fp1")
    assert got is not None and got.hash == "h1" and got.meta["size"] == 3
    assert rs.cache_stats()["entries"] == 1
    rs.close()


def test_runstore_records_runs(tmp_path):
    rs = RunStore(tmp_path / "db.sqlite")
    rs.run_start("r1", label="测试")
    rs.run_end("r1", "success", nodes=3, cached=1)
    runs = rs.runs()
    assert runs[0]["run_id"] == "r1" and runs[0]["status"] == "success"
    assert runs[0]["cached"] == 1
    rs.close()
