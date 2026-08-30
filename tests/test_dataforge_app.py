"""本地服务的接口测试。

界面出问题时，先看这里绿不绿——绿的就是前端的问题，不是执行器的问题。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dataforge.app.server import create_app

FIELDS = [
    {"name": "sensor_id", "type": "string"},
    {"name": "speed", "type": "float", "min": 0, "max": 70, "unit": "mph"},
    {"name": "volume", "type": "integer", "min": 0, "max": 1200},
    {"name": "level", "type": "categorical", "categories": ["low", "mid", "high"]},
]


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "ws")
    with TestClient(app) as c:
        yield c


def wait_done(client, run_id, timeout=30.0):
    """执行在后台线程，轮询到终态。"""
    since = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/runs/{run_id}?since={since}").json()
        since = j["next_since"]
        if j["status"] not in ("running", "pending"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"运行 {run_id} 超时未结束")


def start(client, **over):
    body = {
        "kind": "tabular", "name": "t", "fields": FIELDS,
        "num_records": 200, "seed": 7, "target": "level",
    }
    body.update(over)
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── 基础 ──────────────────────────────────────────────────

def test_health(client):
    j = client.get("/api/health").json()
    assert j["ok"] is True and "cache" in j


def test_index_and_static_served(client):
    assert client.get("/").status_code == 200
    for f in ("style.css", "app.js", "icon.png"):
        assert client.get(f"/static/{f}").status_code == 200, f"{f} 没被服务出来"


def test_defaults_exposed(client):
    j = client.get("/api/defaults").json()
    assert "min_rows" in j["thresholds"]


# ── 运行 ──────────────────────────────────────────────────

def test_run_completes_and_returns_dag(client):
    j = start(client)
    assert j["dag"]["nodes"] and j["dag"]["edges"]
    done = wait_done(client, j["run_id"])
    assert done["status"] == "success", done
    assert set(done["results"]) == {"schema", "generate", "analyze", "gate", "export"}
    assert all(r["status"] == "COMPUTED" for r in done["results"].values())


def test_second_run_hits_cache(client):
    """跨请求的缓存复用 —— 界面上「命中缓存」那一栏靠它。"""
    wait_done(client, start(client)["run_id"])
    done = wait_done(client, start(client)["run_id"])
    assert all(r["status"] == "CACHED" for r in done["results"].values()), done["results"]


def test_changing_rows_only_recomputes_downstream(client):
    wait_done(client, start(client, num_records=200)["run_id"])
    done = wait_done(client, start(client, num_records=400)["run_id"])
    assert done["results"]["schema"]["status"] == "CACHED"
    assert done["results"]["generate"]["status"] == "COMPUTED"


def test_force_recomputes_everything(client):
    wait_done(client, start(client)["run_id"])
    j = start(client, force=["schema", "generate", "analyze", "gate", "export"])
    done = wait_done(client, j["run_id"])
    assert all(r["status"] == "COMPUTED" for r in done["results"].values())


def test_events_stream_incrementally(client):
    """轮询必须是增量的，否则事件会重复堆到界面上。"""
    j = start(client)
    first = client.get(f"/api/runs/{j['run_id']}?since=0").json()
    second = client.get(f"/api/runs/{j['run_id']}?since={first['next_since']}").json()
    assert second["next_since"] >= first["next_since"]
    wait_done(client, j["run_id"])


# ── 阻断与错误 ────────────────────────────────────────────

def test_gate_block_is_reported(client):
    done = wait_done(client, start(client, num_records=5)["run_id"])
    assert done["status"] == "blocked"
    assert done["results"]["gate"]["status"] == "BLOCKED"
    assert done["results"]["export"]["status"] == "SKIPPED"


def test_relaxed_threshold_lets_it_through(client):
    done = wait_done(client, start(client, num_records=5, thresholds={"min_rows": 1})["run_id"])
    assert done["status"] == "success"


def test_bad_pipeline_is_rejected_with_400(client):
    r = client.post("/api/runs", json={
        "kind": "timeseries", "name": "t", "fields": FIELDS, "num_records": 100,
    })
    assert r.status_code == 400 and "time_column" in r.json()["detail"]


def test_invalid_payload_is_422(client):
    r = client.post("/api/runs", json={"kind": "tabular", "fields": FIELDS, "num_records": -5})
    assert r.status_code == 422


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


# ── 产物 ──────────────────────────────────────────────────

def test_dataframe_preview_is_truncated(client):
    done = wait_done(client, start(client, num_records=500)["run_id"])
    ref = done["results"]["generate"]["artifact"]
    j = client.get(f"/api/artifacts/dataframe/{ref['hash']}?limit=50").json()
    assert j["rows"] == 500 and len(j["preview"]) == 50 and j["truncated"] is True
    assert set(j["columns"]) >= {"speed", "volume", "level"}


def test_export_card_marks_synthetic(client):
    """合成标记必须一路带到界面（设计文档 §12 红线一）。"""
    done = wait_done(client, start(client)["run_id"])
    ref = done["results"]["export"]["artifact"]
    card = client.get(f"/api/artifacts/json/{ref['hash']}").json()["value"]
    assert card["synthetic"] is True
    assert "不得用于刷榜" in card["notice"]


def test_gate_report_contains_leakage_section(client):
    done = wait_done(client, start(client)["run_id"])
    ref = done["results"]["gate"]["artifact"]
    rep = client.get(f"/api/artifacts/json/{ref['hash']}").json()["value"]
    assert any(r["gate"] == "leakage" for r in rep["results"])
    assert "audit" in rep


def test_missing_artifact_is_404(client):
    assert client.get("/api/artifacts/json/" + "0" * 32).status_code == 404


# ── 血缘 ──────────────────────────────────────────────────

def test_lineage_endpoint(client):
    j = start(client)
    wait_done(client, j["run_id"])
    lin = client.get(f"/api/runs/{j['run_id']}/lineage").json()
    assert "schema" in lin
    assert lin["schema"]["columns"]["speed"]["unit"] == "mph"


def test_runs_history_lists_finished_run(client):
    j = start(client)
    wait_done(client, j["run_id"])
    runs = client.get("/api/runs").json()["runs"]
    assert any(r["run_id"] == j["run_id"] for r in runs)


# ── 时序 ──────────────────────────────────────────────────

def test_timeseries_pipeline_via_api(client):
    fields = [
        {"name": "ts", "type": "datetime", "start": "2026-01-01", "end": "2026-03-01"},
        {"name": "speed", "type": "float", "min": 0, "max": 70},
    ]
    j = start(client, kind="timeseries", fields=fields, time_column="ts",
              target=None, num_records=300)
    done = wait_done(client, j["run_id"])
    assert done["status"] == "success", done
    assert "split" in done["results"]
