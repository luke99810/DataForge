"""自动调参在接口层的行为。"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dataforge.app.server import create_app

FIELDS = [
    {"name": "speed", "type": "float", "min": 0, "max": 70},
    {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
]


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(tmp_path / "ws")) as c:
        yield c


def wait(client, run_id, timeout=60.0):
    since, deadline = 0, time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/runs/{run_id}?since={since}").json()
        since = j["next_since"]
        if j["status"] not in ("running", "pending"):
            return j
        time.sleep(0.05)
    raise AssertionError("超时")


def start(client, **over):
    body = {"kind": "tabular", "name": "t", "fields": FIELDS,
            "num_records": 500, "seed": 5, "target": "level"}
    body.update(over)
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_auto_refine_off_by_default(client):
    done = wait(client, start(client)["run_id"])
    assert done["loop"] is None, "没勾自动调参就不该跑循环"


def test_auto_refine_recovers_from_blocked_run(client):
    """★ 行数不够被拦下 → 自动加行数 → 通过。整个过程用户不用管。"""
    done = wait(client, start(client, num_records=5, auto_refine=True)["run_id"])

    loop = done["loop"]
    assert loop is not None
    assert loop["reason"] == "success", loop["message"]
    assert len(loop["rounds"]) >= 2
    assert loop["rounds"][0]["status"] == "blocked"
    assert loop["scores"][-1] > loop["scores"][0], "分数要真的涨了"

    changes = loop["rounds"][0]["proposal"]["changes"]
    assert changes[0]["param"] == "num_records"
    assert done["status"] == "success"


def test_loop_returns_real_artifacts_of_best_round(client):
    """界面拿到的必须是最佳那一轮的真实产物，不能是占位。"""
    done = wait(client, start(client, num_records=5, auto_refine=True)["run_id"])
    assert set(done["results"]) == {"schema", "generate", "analyze", "gate", "export"}

    ref = done["results"]["generate"]["artifact"]
    j = client.get(f"/api/artifacts/dataframe/{ref['hash']}?limit=5").json()
    assert j["rows"] >= 50, "最佳轮的数据应当已经满足门禁"
    assert set(j["columns"]) == {"speed", "level"}


def test_loop_never_relaxes_thresholds(client):
    """★ 自动调参绝不降低门禁标准。"""
    done = wait(client, start(
        client, num_records=5, auto_refine=True, thresholds={"min_rows": 200}
    )["run_id"])
    loop = done["loop"]
    for r in loop["rounds"]:
        assert r["params"]["thresholds"] == {"min_rows": 200}, "阈值被改过"

    ref = done["results"]["export"]["artifact"]
    card = client.get(f"/api/artifacts/json/{ref['hash']}").json()["value"]
    assert card["gate_audit"]["relaxed"] is False
    assert card["gate_audit"]["thresholds"]["min_rows"] == 200


def test_loop_respects_max_rounds(client):
    done = wait(client, start(
        client, num_records=5, auto_refine=True, max_rounds=1
    )["run_id"])
    loop = done["loop"]
    assert len(loop["rounds"]) == 1
    assert loop["reason"] in ("exhausted", "blocked")


def test_loop_events_reach_the_ui(client):
    j = start(client, num_records=5, auto_refine=True)
    done = wait(client, j["run_id"])
    kinds = [e.get("kind") for e in done["events"]]
    assert "loop" in kinds, "每轮要推一条事件，否则界面上看不到进度"


def test_max_rounds_out_of_range_is_422(client):
    r = client.post("/api/runs", json={
        "kind": "tabular", "name": "t", "fields": FIELDS,
        "num_records": 100, "auto_refine": True, "max_rounds": 99,
    })
    assert r.status_code == 422
