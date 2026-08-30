"""论文链路的接口测试：上传 → 停在确认 → 回填续跑。"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dataforge.app import server as srv
from dataforge.llm.client import StubLLM

pytest.importorskip("fitz", reason="解析 PDF 需要 PyMuPDF")

from tests.test_dataforge_paper import EXTRACTION, PAPER_TEXT  # noqa: E402


@pytest.fixture()
def pdf_bytes(tmp_path):
    import fitz

    doc = fitz.open()
    for body in PAPER_TEXT:
        doc.new_page().insert_text((60, 80), body, fontsize=10)
    p = tmp_path / "p.pdf"
    doc.save(p)
    doc.close()
    return p.read_bytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 桩模型：测试不该依赖网络，也不该因模型今天心情不同而变红
    monkeypatch.setattr(srv, "from_env", lambda: StubLLM(default=EXTRACTION))
    app = srv.create_app(tmp_path / "ws")
    with TestClient(app) as c:
        yield c


def wait(client, run_id, timeout=30.0):
    since, deadline = 0, time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/runs/{run_id}?since={since}").json()
        since = j["next_since"]
        if j["status"] not in ("running", "pending"):
            return j
        time.sleep(0.05)
    raise AssertionError("超时")


def upload(client, data, name="paper.pdf"):
    r = client.post("/api/papers", files={"file": (name, data, "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()


def start(client, pid, decisions=None, **over):
    body = {"kind": "paper", "paper_id": pid, "name": "p", "num_records": 200, "seed": 3}
    if decisions is not None:
        body["decisions"] = decisions
    body.update(over)
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── 上传 ──────────────────────────────────────────────────

def test_upload_returns_content_addressed_id(client, pdf_bytes):
    a = upload(client, pdf_bytes)
    b = upload(client, pdf_bytes, name="别的名字.pdf")
    assert a["paper_id"] == b["paper_id"], "同一份内容应得到同一个 id，缓存才命中得了"
    assert a["size"] == len(pdf_bytes)


def test_upload_rejects_non_pdf(client):
    r = client.post("/api/papers", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_llm_status_reports_configured(client):
    j = client.get("/api/llm").json()
    assert j["configured"] is True


# ── 停在确认 ──────────────────────────────────────────────

def test_paper_run_stops_for_confirmation(client, pdf_bytes):
    pid = upload(client, pdf_bytes)["paper_id"]
    done = wait(client, start(client, pid)["run_id"])

    assert done["status"] == "needs_human"
    assert done["results"]["parse"]["status"] == "COMPUTED"
    assert done["results"]["extract"]["status"] == "COMPUTED"

    ctx = done["ask"]["context"]
    names = [v["name"] for v in ctx["variables"]]
    assert names == ["speed", "volume", "occupancy", "rainfall"]
    assert set(ctx["suggested"]) == {"speed", "volume", "occupancy"}
    by = {v["name"]: v for v in ctx["variables"]}
    assert by["rainfall"]["from_related_work"] is True
    assert by["speed"]["source_span"]["quote"]
    assert ctx["open_questions"], "论文没写清的地方要交给人"


def test_missing_paper_id_is_400(client):
    r = client.post("/api/runs", json={"kind": "paper", "name": "p", "num_records": 10})
    assert r.status_code == 400 and "paper_id" in r.json()["detail"]


def test_unknown_paper_id_is_400(client):
    r = client.post("/api/runs", json={"kind": "paper", "paper_id": "deadbeef",
                                       "name": "p", "num_records": 10})
    assert r.status_code == 400 and "找不到论文" in r.json()["detail"]


# ── ★ 回填续跑 ────────────────────────────────────────────

def test_confirmation_resumes_and_reuses_cache(client, pdf_bytes):
    """★ 界面上点「确认并继续」之后：上游命中缓存，不重复调模型。"""
    pid = upload(client, pdf_bytes)["paper_id"]
    wait(client, start(client, pid)["run_id"])

    done = wait(client, start(client, pid, decisions={
        "adopt": ["speed", "volume", "occupancy"],
        "answers": {"论文未给出原始缺失率，用多少？": "5%"},
    })["run_id"])

    assert done["status"] == "success", done
    assert done["results"]["parse"]["status"] == "CACHED"
    assert done["results"]["extract"]["status"] == "CACHED", "抽取是花钱那步，不该重跑"
    assert done["results"]["confirm"]["status"] == "COMPUTED"
    assert done["results"]["export"]["status"] == "COMPUTED"


def test_resumed_dataset_only_has_adopted_columns(client, pdf_bytes):
    pid = upload(client, pdf_bytes)["paper_id"]
    wait(client, start(client, pid)["run_id"])
    done = wait(client, start(client, pid, decisions={"adopt": ["speed", "volume"]})["run_id"])

    ref = done["results"]["generate"]["artifact"]
    j = client.get(f"/api/artifacts/dataframe/{ref['hash']}?limit=5").json()
    assert j["columns"] == ["speed", "volume"]
    assert "rainfall" not in j["columns"], "没采纳的变量不能出现在数据里"


def test_human_answers_land_in_lineage(client, pdf_bytes):
    pid = upload(client, pdf_bytes)["paper_id"]
    wait(client, start(client, pid)["run_id"])
    j = start(client, pid, decisions={
        "adopt": ["speed"],
        "overrides": {"speed": {"unit": "km/h"}},
        "answers": {"speed 的单位？": "km/h"},
    })
    wait(client, j["run_id"])

    lin = client.get(f"/api/runs/{j['run_id']}/lineage").json()
    decisions = lin["confirm"]["decisions"]
    assert any(d["answer"] == "km/h" for d in decisions)
    assert lin["confirm"]["columns"]["speed"]["origin"] == "paper"
    assert lin["confirm"]["columns"]["speed"]["source_span"]["page"] == 3


def test_static_paper_js_is_served(client):
    assert client.get("/static/paper.js").status_code == 200


def test_no_llm_configured_reports_clearly(tmp_path, monkeypatch, pdf_bytes):
    """没配模型时明确失败，不悄悄退化。"""
    monkeypatch.setattr(srv, "from_env", lambda: None)
    app = srv.create_app(tmp_path / "ws2")
    with TestClient(app) as c:
        assert c.get("/api/llm").json()["configured"] is False
        pid = upload(c, pdf_bytes)["paper_id"]
        done = wait(c, start(c, pid)["run_id"])
        assert done["status"] == "failed"
        assert "没有配置模型" in done["results"]["extract"]["error"]
