"""外部源在 MCP 层的行为。

调用方是另一个 Agent，所以「默认不出网」「合成声明不可关闭」
这两条必须在协议层堵住，不能靠界面。
"""

from __future__ import annotations

import pytest

from dataforge import mcp_server as M
from dataforge.llm.client import StubLLM
from dataforge.sources import StubFetcher

from tests.test_dataforge_planner import GOOD
from tests.test_dataforge_sources import HF_INFO


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "from_env", lambda: StubLLM(default=GOOD))
    w = M.Workspace(tmp_path / "ws")
    # 换成桩取数器：测试不出网
    w.fetcher = StubFetcher({"datasets-server": HF_INFO})
    yield w
    w.close()


def test_tools_registered():
    names = {t["name"] for t in M.TOOLS}
    assert {"df_inspect_source", "df_align_to_source", "df_list_benchmarks"} <= names


def test_list_benchmarks(ws):
    out = M.call_tool(ws, "df_list_benchmarks", {})
    keys = {b["key"] for b in out["benchmarks"]}
    assert {"gsm8k", "mmlu", "humaneval"} <= keys
    for b in out["benchmarks"]:
        assert b["protocol"].get("metric")
    assert "必然滞后" in out["note"]


def test_inspect_registry_hit_without_network(ws):
    out = M.call_tool(ws, "df_inspect_source", {"link": "gsm8k"})
    assert out["resolved_by"] == "registry"
    assert out["benchmark"]["name"] == "GSM8K"
    assert [f["name"] for f in out["fields"]] == ["question", "answer"]
    assert ws.fetcher.calls == [], "命中注册表还出网了"
    assert "没有下载任何数据行" in out["hint"]


def test_inspect_does_not_generate_anything(ws):
    """★ inspect 只看结构，不产出数据。"""
    M.call_tool(ws, "df_inspect_source", {"link": "gsm8k"})
    assert M.call_tool(ws, "df_list_runs", {})["runs"] == []


def test_network_off_by_default(ws):
    """★ 出网必须显式开。"""
    out = M.call_tool(ws, "df_align_to_source", {"link": "hf://datasets/demo/ds"})
    assert out["status"] == "failed"
    assert "显式传入 fetcher" in out["nodes"]["source"]["error"]
    assert ws.fetcher.calls == []


def test_align_to_benchmark_marks_synthetic(ws):
    """★ 合成声明与来源必须一路带到产出。"""
    out = M.call_tool(ws, "df_align_to_source",
                      {"link": "imdb", "num_records": 200, "name": "imdb_like"})
    assert out["status"] == "success"
    assert out["aligned_to"]["benchmark"] == "IMDB"
    assert "不得用于刷榜" in out["notice"]

    card = M.call_tool(ws, "df_fetch_artifact", {
        "hash": out["nodes"]["export"]["artifact"]["hash"], "kind": "json"})["value"]
    assert card["synthetic"] is True
    assert card["provenance"]["source"] == "builtin_registry"


def test_align_with_network_reports_skipped_fields(ws):
    """★ 映射不了的字段要明确警告，否则用户以为数据是完整的。"""
    out = M.call_tool(ws, "df_align_to_source", {
        "link": "hf://datasets/demo/ds", "num_records": 100, "allow_network": True,
    })
    assert out["status"] == "success"
    assert set(out["aligned_to"]["skipped_fields"]) == {"embedding", "image"}
    assert "映射不了" in out["warning"]
    assert ws.fetcher.calls, "开了网就该真去取元数据"


def test_unknown_link_says_it_cannot_recognise(ws):
    """★ 认不出来就直说认不出来，别拿「注册表里没有」去误导。"""
    with pytest.raises(Exception, match="认不出"):
        M.call_tool(ws, "df_inspect_source", {"link": "完全不认识的东西"})


def test_known_kind_but_needs_network_says_that_instead(ws):
    """认得出是 HF 数据集、只是没开网 —— 错误信息要指向这一点。"""
    with pytest.raises(Exception, match="显式传入 fetcher"):
        M.call_tool(ws, "df_inspect_source", {"link": "hf://datasets/someone/unseen"})


def test_exported_dataset_can_be_written_with_datacard(ws, tmp_path):
    gen = M.call_tool(ws, "df_align_to_source", {"link": "gsm8k", "num_records": 120})
    out = M.call_tool(ws, "df_export", {
        "hash": gen["nodes"]["generate"]["artifact"]["hash"],
        "out_path": str(tmp_path / "gsm8k_like.csv"),
    })
    import json

    card = json.loads((tmp_path / "gsm8k_like.csv.datacard.json").read_text(encoding="utf-8"))
    assert card["synthetic"] is True
    assert out["rows"] == 120
