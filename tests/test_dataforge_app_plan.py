"""Planner 在接口层的行为：只规划，不执行。"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dataforge.app import server as srv
from dataforge.llm.client import StubLLM

from tests.test_dataforge_planner import GOOD, spec


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "from_env", lambda: StubLLM(default=GOOD))
    with TestClient(srv.create_app(tmp_path / "ws")) as c:
        yield c


def wait(client, run_id, timeout=40.0):
    since, deadline = 0, time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/runs/{run_id}?since={since}").json()
        since = j["next_since"]
        if j["status"] not in ("running", "pending"):
            return j
        time.sleep(0.05)
    raise AssertionError("超时")


def test_plan_returns_form_request(client):
    r = client.post("/api/plan", json={"text": "交通拥堵分类数据集"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["plan"]["spec"]["name"] == "traffic_flow"
    req = j["request"]
    assert req["kind"] == "tabular" and req["target"] == "congestion"
    assert [f["name"] for f in req["fields"]] == \
        ["road_id", "speed", "volume", "occupancy", "congestion"]


def test_plan_does_not_execute_anything(client):
    """★ 规划只填表单，绝不顺手跑一遍——跑不跑由用户决定。"""
    client.post("/api/plan", json={"text": "交通拥堵分类数据集"})
    assert client.get("/api/runs").json()["runs"] == []
    assert client.get("/api/health").json()["cache"]["entries"] <= 1  # 只有计划书本身


def test_planned_request_actually_runs(client):
    """★ 规划出来的请求体能原样跑通 —— 这才叫接上了。"""
    req = client.post("/api/plan", json={"text": "交通拥堵分类数据集"}).json()["request"]
    started = client.post("/api/runs", json=req)
    assert started.status_code == 200, started.text
    done = wait(client, started.json()["run_id"])
    assert done["status"] == "success", done

    ref = done["results"]["generate"]["artifact"]
    d = client.get(f"/api/artifacts/dataframe/{ref['hash']}?limit=3").json()
    assert d["rows"] == 800
    assert set(d["columns"]) == {"road_id", "speed", "volume", "occupancy", "congestion"}


def test_plan_is_cached_across_requests(client):
    a = client.post("/api/plan", json={"text": "同一段需求"}).json()["plan"]
    b = client.post("/api/plan", json={"text": "同一段需求"}).json()["plan"]
    assert a["cached"] is False and b["cached"] is True


def test_bad_plan_returns_plan_with_issues(client, monkeypatch):
    """模型给了一份说不通的计划书 —— 200 带着计划书和问题清单。

    不用 400：那样只剩一句错误，用户看不到模型到底产出了什么、
    也就不知道该怎么改需求描述。
    """
    bad = spec(target="ghost")
    for f in bad["fields"]:
        f["role"] = "target"          # 多个候选 → autofix 无法纠正，只能清空
    bad["fields"][0]["name"] = "道路编号"   # 中文字段名，autofix 不碰
    monkeypatch.setattr(srv, "from_env", lambda: StubLLM(default=bad))
    with TestClient(srv.create_app("./.dfplan_bad")) as c:
        r = c.post("/api/plan", json={"text": "随便"})
        assert r.status_code == 200
        j = r.json()
        assert j["request"] is None, "跑不通的计划不该给出可执行的请求体"
        assert any("合法标识符" in i for i in j["plan"]["issues"])
        assert j["plan"]["spec"]["fields"], "但计划书本身要摆出来给人看"


def test_empty_text_is_422(client):
    assert client.post("/api/plan", json={"text": ""}).status_code == 422


def test_no_llm_returns_400_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "from_env", lambda: None)
    with TestClient(srv.create_app(tmp_path / "ws2")) as c:
        r = c.post("/api/plan", json={"text": "来点数据"})
        assert r.status_code == 400
        assert "没有配置模型" in r.json()["detail"]


def test_timeseries_plan_routes_to_timeseries_pipeline(client, monkeypatch):
    s = spec(pipeline="timeseries", task_type="time_series_forecast", time_column="ts")
    s["fields"].insert(0, {"name": "ts", "dtype": "datetime",
                           "start": "2026-01-01", "end": "2026-02-01"})
    monkeypatch.setattr(srv, "from_env", lambda: StubLLM(default=s))
    with TestClient(srv.create_app("./.dfplan_ts")) as c:
        req = c.post("/api/plan", json={"text": "交通流预测"}).json()["request"]
        assert req["kind"] == "timeseries" and req["time_column"] == "ts"
        done = wait(c, c.post("/api/runs", json=req).json()["run_id"])
        assert done["status"] == "success", done
        assert "split" in done["results"], "时序流水线必须有按时间切分的节点"
