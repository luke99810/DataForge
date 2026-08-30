"""新三类生成器在 MCP 层的行为。"""

from __future__ import annotations

import pytest

from dataforge import mcp_server as M
from dataforge.llm.client import StubLLM

from tests.test_dataforge_planner import GOOD


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "from_env", lambda: StubLLM(default=GOOD))
    w = M.Workspace(tmp_path / "ws")
    yield w
    w.close()


def test_new_tools_registered():
    names = {t["name"] for t in M.TOOLS}
    assert {"df_rl_dataset", "df_agent_eval_set",
            "df_llm_dataset", "df_run_assertions"} <= names


def test_rl_dataset_over_mcp(ws):
    out = M.call_tool(ws, "df_rl_dataset", {"env": "gridworld", "num_episodes": 40})
    assert out["status"] == "success"
    assert out["env_spec"]["env"] == "gridworld"
    assert out["trajectory_stats"]["num_episodes"] == 40
    assert "重放核对" in out["hint"]


def test_rl_rejects_bad_epsilon(ws):
    out = M.call_tool(ws, "df_rl_dataset", {"epsilon": 5.0, "num_episodes": 5})
    assert out["status"] == "failed"


def test_agent_eval_set_over_mcp(ws):
    out = M.call_tool(ws, "df_agent_eval_set", {"num_tasks": 12, "categories": ["compute"]})
    assert out["status"] == "success"
    assert out["eval_stats"]["num_tasks"] == 12
    assert out["sample_task"]["assertions"], "每个任务必须带断言"
    assert "LLM-as-Judge" in out["hint"]


def test_run_assertions_over_mcp(ws):
    """★ 判对错的是代码：同一个任务，对的通过、错的不通过。"""
    gen = M.call_tool(ws, "df_agent_eval_set", {"num_tasks": 1, "categories": ["compute"]})
    task = gen["sample_task"]
    expected = next(a["value"] for a in task["assertions"] if a["kind"] == "value_equals")

    good = M.call_tool(ws, "df_run_assertions", {
        "task": task,
        "result": {"output": "", "result": expected,
                   "tool_calls": [{"name": "calculator"}], "error": None},
    })
    bad = M.call_tool(ws, "df_run_assertions", {
        "task": task,
        "result": {"output": "", "result": expected + 1, "tool_calls": [], "error": None},
    })
    assert good["passed"] is True
    assert bad["passed"] is False and bad["failed"]


def test_llm_dataset_over_mcp(ws):
    out = M.call_tool(ws, "df_llm_dataset", {"num_samples": 60})
    assert out["status"] == "success"
    assert out["sft_stats"]["kept"] > 0
    assert out["dpo_stats"]["num_pairs"] > 0
    assert "污染" in out["hint"]


def test_llm_dataset_blocks_on_caller_supplied_contamination(ws):
    """★ 调用方传进来的评测集也要真的用上 —— 命中就必须阻断。"""
    out = M.call_tool(ws, "df_llm_dataset", {
        "num_samples": 20,
        # 直接拿生成器模板里的句式当"评测题"，必然命中
        "reference": [["MyBench", "什么是过拟合？用一两句话说清楚。"]],
    })
    assert out["status"] == "blocked"
    assert any(g["gate"] == "contamination" and not g["passed"]
               for g in out["gate_results"])


def test_rl_and_eval_results_are_cached(ws):
    a = M.call_tool(ws, "df_rl_dataset", {"num_episodes": 20})
    b = M.call_tool(ws, "df_rl_dataset", {"num_episodes": 20})
    assert all(n["status"] == "CACHED" for n in b["nodes"].values())
    assert a["nodes"]["generate"]["artifact"]["hash"] == \
        b["nodes"]["generate"]["artifact"]["hash"]
