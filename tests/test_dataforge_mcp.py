"""MCP 服务端。

重点在**红线对外部 Agent 同样成立**——调用方是另一个 Agent，
不会像人那样看一眼再点确认，所以闸门必须在协议层就堵住。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataforge import mcp_server as M
from dataforge.llm.client import StubLLM

pytest.importorskip("fitz", reason="论文相关用例需要 PyMuPDF")

from tests.test_dataforge_paper import EXTRACTION, PAPER_TEXT  # noqa: E402
from tests.test_dataforge_planner import GOOD  # noqa: E402

FIELDS = [
    {"name": "speed", "type": "float", "min": 0, "max": 70, "unit": "mph"},
    {"name": "volume", "type": "integer", "min": 0, "max": 1200},
    {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
]


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "from_env", lambda: StubLLM(default=GOOD))
    w = M.Workspace(tmp_path / "ws")
    yield w
    w.close()


@pytest.fixture()
def pdf(tmp_path):
    import fitz

    doc = fitz.open()
    for body in PAPER_TEXT:
        doc.new_page().insert_text((60, 80), body, fontsize=10)
    p = tmp_path / "paper.pdf"
    doc.save(p)
    doc.close()
    return p


# ── 注册表 ────────────────────────────────────────────────

def test_all_tools_have_schema_and_handler():
    for t in M.TOOLS:
        assert t["name"].startswith("df_")
        assert t["description"] and callable(t["handler"])
        assert t["schema"]["type"] == "object"
        for req in t["schema"].get("required", []):
            assert req in t["schema"]["properties"], f"{t['name']} 的必填 {req} 没定义"


def test_tool_names_are_unique():
    names = [t["name"] for t in M.TOOLS]
    assert len(names) == len(set(names))


def test_unknown_tool_is_rejected(ws):
    with pytest.raises(ValueError, match="未知工具"):
        M.call_tool(ws, "df_nope", {})


# ── 生成 ──────────────────────────────────────────────────

def test_generate_returns_summary_not_data(ws):
    """不回传数据本体 —— 几十万行塞进对话上下文毫无意义。"""
    out = M.call_tool(ws, "df_generate", {"fields": FIELDS, "num_records": 300,
                                          "target": "level", "name": "t"})
    assert out["status"] == "success"
    assert out["quality_score"] == 100.0
    assert out["gate_passed"] is True
    assert "preview" not in out and "data" not in out
    assert out["nodes"]["generate"]["artifact"]["kind"] == "dataframe"
    assert "不得用于刷榜" in out["notice"]


def test_generate_reports_block_without_crashing(ws):
    out = M.call_tool(ws, "df_generate", {"fields": FIELDS, "num_records": 5, "name": "t"})
    assert out["status"] == "blocked"
    assert out["gate_passed"] is False
    assert any(g["gate"] == "row_count" and not g["passed"] for g in out["gate_results"])
    assert out["quality_score"] < 100


def test_generate_requires_fields(ws):
    with pytest.raises(ValueError, match="至少需要一个字段"):
        M.call_tool(ws, "df_generate", {"fields": []})


def test_timeseries_requires_time_column(ws):
    with pytest.raises(ValueError, match="time_column"):
        M.call_tool(ws, "df_generate", {"fields": FIELDS, "kind": "timeseries"})


# ── ★ 红线 ────────────────────────────────────────────────

def test_thresholds_cannot_be_set_by_caller(ws):
    """★ 阈值不接受外部指定。

    允许调用方传阈值，等于把「把标准降到自己够得着为止」这条路重新打开。
    """
    assert "thresholds" not in M.BY_NAME["df_generate"]["schema"]["properties"]

    out = M.call_tool(ws, "df_generate", {
        "fields": FIELDS, "num_records": 5, "name": "t",
        "thresholds": {"min_rows": 1},        # 就算硬传也不该生效
    })
    assert out["status"] == "blocked", "阈值被外部改动了"


def test_paper_tool_stops_at_variable_cards(ws, pdf, monkeypatch):
    """★ 人工确认闸门对 MCP 调用方同样成立。"""
    monkeypatch.setattr(M, "from_env", lambda: StubLLM(default=EXTRACTION))
    w = M.Workspace(ws.root)
    try:
        out = M.call_tool(w, "df_parse_paper", {"path": str(pdf)})
        assert [v["name"] for v in out["variables"]] == \
            ["speed", "volume", "occupancy", "rainfall"]
        assert out["suggested_adopt"] == ["speed", "volume", "occupancy"]
        assert "run_id" not in out, "解析阶段不该产出数据集"
        assert "人工确认闸门" in out["gate"]
        assert "source_span" in out["variables"][0]
    finally:
        w.close()


def test_dataset_from_paper_requires_explicit_adopt(ws, pdf, monkeypatch):
    """★ 拿不到「一句话直接出数据集」的捷径。"""
    monkeypatch.setattr(M, "from_env", lambda: StubLLM(default=EXTRACTION))
    w = M.Workspace(ws.root)
    try:
        with pytest.raises(ValueError, match="必须显式列出采纳的变量"):
            M.call_tool(w, "df_dataset_from_paper", {"path": str(pdf)})
        with pytest.raises(ValueError, match="必须显式列出"):
            M.call_tool(w, "df_dataset_from_paper", {"path": str(pdf), "adopt": []})
    finally:
        w.close()


def test_dataset_from_paper_honours_adopt(ws, pdf, monkeypatch):
    monkeypatch.setattr(M, "from_env", lambda: StubLLM(default=EXTRACTION))
    w = M.Workspace(ws.root)
    try:
        out = M.call_tool(w, "df_dataset_from_paper", {
            "path": str(pdf), "adopt": ["speed", "volume"], "num_records": 120,
        })
        assert out["status"] == "success"
        assert out["adopted"] == ["speed", "volume"]
        assert out["provenance"]["speed"]["source_span"]["page"] == 3

        df = M.call_tool(w, "df_fetch_artifact", {
            "hash": out["nodes"]["generate"]["artifact"]["hash"],
            "kind": "dataframe", "limit": 5,
        })
        assert df["columns"] == ["speed", "volume"]
        assert "rainfall" not in df["columns"]
    finally:
        w.close()


def test_export_always_writes_datacard(ws, tmp_path):
    """★ 数据卡与数据同去同回：只给数据不给卡，下游就无从判断它是合成的。"""
    gen = M.call_tool(ws, "df_generate", {"fields": FIELDS, "num_records": 200, "name": "t"})
    out = M.call_tool(ws, "df_export", {
        "hash": gen["nodes"]["generate"]["artifact"]["hash"],
        "out_path": str(tmp_path / "out" / "data.csv"),
    })
    card = json.loads((tmp_path / "out" / "data.csv.datacard.json").read_text(encoding="utf-8"))
    assert card["synthetic"] is True
    assert "不得用于刷榜" in card["notice"]
    assert pd.read_csv(out["path"]).shape[0] == 200


# ── 校验与分析 ────────────────────────────────────────────

def test_validate_detects_leakage(ws, tmp_path):
    import numpy as np

    rng = np.random.default_rng(1)
    y = rng.normal(50, 10, 400)
    p = tmp_path / "leaky.csv"
    pd.DataFrame({"noise": rng.normal(0, 1, 400),
                  "sneaky": y * 1.0001, "target": y}).to_csv(p, index=False)

    out = M.call_tool(ws, "df_validate", {"path": str(p), "target": "target"})
    assert out["has_error"] is True
    assert any(f["kind"] == "target" for f in out["leakage"])
    assert "静默故障" in out["hint"]


def test_validate_respects_known_correlated(ws, tmp_path):
    import numpy as np

    rng = np.random.default_rng(2)
    base = rng.normal(100, 20, 400)
    p = tmp_path / "physics.csv"
    pd.DataFrame({"volume": base, "occupancy": base * 0.9 + rng.normal(0, 3, 400)}).to_csv(p, index=False)

    out = M.call_tool(ws, "df_validate", {
        "path": str(p), "target": "occupancy",
        "known_correlated": [["volume", "occupancy"]],
    })
    assert out["has_error"] is False, "物理规律导致的高相关不该报成泄漏"


def test_validate_rejects_unknown_format(ws, tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的格式"):
        M.call_tool(ws, "df_validate", {"path": str(p)})


def test_analyze_returns_stats(ws, tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": range(200), "b": [1.5] * 200}).to_csv(p, index=False)
    out = M.call_tool(ws, "df_analyze", {"path": str(p)})
    assert out["rows"] == 200
    assert "a" in out["describe"] and "histograms" in out


def test_missing_file_raises(ws):
    with pytest.raises(FileNotFoundError):
        M.call_tool(ws, "df_validate", {"path": "nope.csv"})


# ── 规划 ──────────────────────────────────────────────────

def test_plan_tool_does_not_execute(ws):
    """★ 规划只规划。"""
    out = M.call_tool(ws, "df_plan", {"text": "交通拥堵分类数据集"})
    assert out["plan"]["name"] == "traffic_flow"
    assert out["runnable_args"]["kind"] == "tabular"
    assert M.call_tool(ws, "df_list_runs", {})["runs"] == []
    assert "检查无误后" in out["hint"]


def test_plan_output_is_directly_runnable(ws):
    """★ 规划产出的 runnable_args 能原样喂给 df_generate。"""
    plan = M.call_tool(ws, "df_plan", {"text": "交通拥堵分类数据集"})
    out = M.call_tool(ws, "df_generate", plan["runnable_args"])
    assert out["status"] == "success"


# ── 运行记录与血缘 ────────────────────────────────────────

def test_runs_and_lineage(ws):
    gen = M.call_tool(ws, "df_generate", {"fields": FIELDS, "num_records": 150, "name": "t"})
    runs = M.call_tool(ws, "df_list_runs", {})["runs"]
    assert any(r["run_id"] == gen["run_id"] for r in runs)

    lin = M.call_tool(ws, "df_lineage", {"run_id": gen["run_id"]})
    assert "schema" in lin and lin["schema"]["columns"]["speed"]["unit"] == "mph"


def test_lineage_missing_run_raises(ws):
    with pytest.raises(FileNotFoundError):
        M.call_tool(ws, "df_lineage", {"run_id": "nope"})


def test_second_identical_generate_hits_cache(ws):
    a = M.call_tool(ws, "df_generate", {"fields": FIELDS, "num_records": 200, "name": "t"})
    b = M.call_tool(ws, "df_generate", {"fields": FIELDS, "num_records": 200, "name": "t"})
    assert a["nodes"]["generate"]["artifact"]["hash"] == b["nodes"]["generate"]["artifact"]["hash"]
    assert all(n["status"] == "CACHED" for n in b["nodes"].values())
