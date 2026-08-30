"""本地 HTTP 服务 —— 桌面界面与执行器之间的桥。

只监听回环地址。数据默认不出机器（设计文档 §10「本地优先」）。

运行是**后台线程 + 轮询**，不是 SSE：本地进程内轮询的延迟只有几十毫秒，
而 SSE 在 pywebview 的 WebView2 里断线重连的边界情况更多。等真需要
跨进程推送时再换。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.artifacts import ArtifactRef, ArtifactStore
from ..core.runstore import RunStore
from ..graph.executor import Executor
from ..harness.budget import Budget, PlateauDetector
from ..harness.loop import RefineLoop
from ..harness.refine import LLMRefiner
from ..llm.client import from_env
from ..planner import PlanError, Planner
from ..pipelines import paper_pipeline, tabular_pipeline, timeseries_pipeline

WEB_DIR = Path(__file__).parent / "web"


# ── 运行状态（内存，仅当前进程） ──────────────────────────

@dataclass
class LiveRun:
    run_id: str
    status: str = "pending"
    events: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    dag: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    ask: dict[str, Any] | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    loop: dict[str, Any] | None = None      # 自动调参的逐轮记录
    lock: threading.Lock = field(default_factory=threading.Lock)


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.root / "artifacts")
        # SQLite 默认禁止跨线程复用连接；执行在后台线程，所以放开检查并自己加锁
        self.runstore = RunStore(self.root / "runs.sqlite")
        self.runstore.conn.execute("PRAGMA journal_mode=WAL")
        self.live: dict[str, LiveRun] = {}
        self._lock = threading.Lock()
        self.papers = self.root / "papers"
        self.papers.mkdir(parents=True, exist_ok=True)
        # L3 是唯一构造 LLM 客户端的地方（设计文档 §2.1）。
        # 没配密钥时为 None，界面据此提示，而不是悄悄退化。
        self.llm = from_env()


# ── 请求体 ────────────────────────────────────────────────

class FieldSpec(BaseModel):
    name: str
    type: str = "float"
    description: str = ""
    min: float | None = None
    max: float | None = None
    categories: list[str] | None = None
    unit: str | None = None
    start: str | None = None
    end: str | None = None


class PlanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    seed: int = 20260823


class Decisions(BaseModel):
    adopt: list[str] | None = None
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    answers: dict[str, str] = Field(default_factory=dict)
    task_type: str | None = None


class RunRequest(BaseModel):
    kind: str = Field(default="tabular", pattern="^(tabular|timeseries|paper)$")
    paper_id: str | None = None
    decisions: Decisions | None = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    name: str = "dataset"
    fields: list[FieldSpec] = Field(default_factory=list)
    num_records: int = Field(default=500, ge=1, le=2_000_000)
    seed: int = 20260823
    target: str | None = None
    task_type: str = "classification"
    time_column: str | None = None
    missing_rate: float = Field(default=0.0, ge=0.0, le=0.95)
    noise_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    thresholds: dict[str, float] = Field(default_factory=dict)
    known_correlated: list[list[str]] = Field(default_factory=list)
    force: list[str] = Field(default_factory=list)
    # 自动调参：不达标时按规则调参重试，而不是把失败直接甩给用户
    auto_refine: bool = False
    max_rounds: int = Field(default=5, ge=1, le=20)


def build_dag(req: RunRequest, papers_dir: Path | None = None):
    if req.kind == "paper":
        if not req.paper_id:
            raise ValueError("论文流水线必须先上传 PDF 并传 paper_id")
        pdf = (papers_dir or Path(".")) / f"{req.paper_id}.pdf"
        if not pdf.exists():
            raise ValueError(f"找不到论文 {req.paper_id}（可能已被清理）")
        return paper_pipeline(
            pdf_path=str(pdf), name=req.name, num_records=req.num_records,
            decisions=req.decisions.model_dump(exclude_none=True) if req.decisions else None,
            min_confidence=req.min_confidence, thresholds=req.thresholds,
        )
    fields = [f.model_dump(exclude_none=True) for f in req.fields]
    if not fields:
        raise ValueError("至少需要一个字段")
    if req.kind == "timeseries":
        if not req.time_column:
            raise ValueError("时序流水线必须指定 time_column")
        return timeseries_pipeline(
            fields=fields, time_column=req.time_column, num_records=req.num_records,
            name=req.name, target=req.target, thresholds=req.thresholds,
            known_correlated=req.known_correlated,
        )
    return tabular_pipeline(
        fields=fields, num_records=req.num_records, name=req.name, target=req.target,
        task_type=req.task_type, missing_rate=req.missing_rate, noise_ratio=req.noise_ratio,
        thresholds=req.thresholds, known_correlated=req.known_correlated,
    )


def _run_loop(ex, req: "RunRequest", live: LiveRun, run_id: str, on_event) -> Any:
    """自动调参模式。

    每一轮都是一次完整的图执行，轮与轮之间共享指纹缓存——
    只改了 num_records 的话，schema 不会重算。
    """
    def build(params: dict[str, Any]):
        r = req.model_copy(update={
            "num_records": int(params.get("num_records", req.num_records)),
            "missing_rate": float(params.get("missing_rate", req.missing_rate)),
            "noise_ratio": float(params.get("noise_ratio", req.noise_ratio)),
        })
        return build_dag(r)

    last: dict[str, Any] = {}

    def on_round(rnd) -> None:
        last["rep"] = rnd
        with live.lock:
            live.events.append({
                "kind": "loop", "node_id": None,
                "round": rnd.index, "score": round(rnd.score, 2),
                "status": rnd.status,
                "message": (rnd.proposal or {}).get("rationale", ""),
            })

    loop = RefineLoop(
        ex, build,
        refiner=LLMRefiner(ex.llm) if ex.llm is not None else None,
        plateau=PlateauDetector(epsilon=0.5, patience=2),
        budget=Budget(max_rounds=req.max_rounds),
        on_round=on_round,
    )
    report = loop.run(
        {"num_records": req.num_records, "missing_rate": req.missing_rate,
         "noise_ratio": req.noise_ratio, "thresholds": req.thresholds},
        seed=req.seed, label=req.name,
    )
    with live.lock:
        live.loop = report.to_dict()

    # 把最佳那一轮的**真实**运行结果返回给界面：
    # 图谱、质量、数据预览用的都是它，不能是占位对象
    best = report.best
    if best is None or best.report is None:
        raise RuntimeError("循环一轮都没跑起来")
    return best.report


# ── 应用 ──────────────────────────────────────────────────

def create_app(workspace: str | Path = "./.dataforge") -> FastAPI:
    ws = Workspace(workspace)
    app = FastAPI(title="DataForge", version="0.1.0")
    app.state.ws = ws

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "workspace": str(ws.root),
            "cache": ws.runstore.cache_stats(),
        }

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict[str, Any]:
        try:
            dag = build_dag(req, ws.papers)
            dag.validate()
        except Exception as e:
            raise HTTPException(400, f"流水线构建失败：{e}") from e

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        live = LiveRun(run_id=run_id, status="running", dag=dag.to_dict())
        with ws._lock:
            ws.live[run_id] = live

        def on_event(kind: str, data: dict[str, Any]) -> None:
            with live.lock:
                live.events.append({"kind": kind, **data})

        def work() -> None:
            # 后台线程用独立的 RunStore 连接，避免 sqlite 跨线程报错
            rs = RunStore(ws.root / "runs.sqlite")
            try:
                ex = Executor(ws.store, rs, workdir=ws.root,
                              on_event=on_event, llm=ws.llm)
                if req.auto_refine:
                    rep = _run_loop(ex, req, live, run_id, on_event)
                else:
                    rep = ex.run(dag, run_id=run_id, seed=req.seed,
                                 force=req.force, label=req.name)
                with live.lock:
                    live.status = rep.status
                    live.results = {
                        nid: {
                            "status": r.status.value,
                            "kind": r.kind,
                            "fingerprint": r.fingerprint,
                            "artifact": r.artifact.to_dict() if r.artifact else None,
                            "duration": round(r.duration, 4),
                            "error": r.error,
                        }
                        for nid, r in rep.results.items()
                    }
                    live.lineage = rep.lineage.to_dict()
                    # 不再把整条轨迹补一遍：on_event 已经逐条推过了，
                    # 补发会让界面上每个节点出现两次
                    asks = [e for e in rep.trace if e.kind == "ask"]
                    if asks:
                        live.ask = asks[-1].data
            except Exception as e:  # noqa: BLE001
                with live.lock:
                    live.status = "error"
                    live.error = f"{type(e).__name__}: {e}"
                    live.events.append({"kind": "node_error",
                                        "error": traceback.format_exc(limit=5)})
            finally:
                rs.close()

        threading.Thread(target=work, daemon=True, name=f"run-{run_id}").start()
        return {"run_id": run_id, "dag": live.dag}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, since: int = 0) -> dict[str, Any]:
        live = ws.live.get(run_id)
        if live is None:
            raise HTTPException(404, "运行不存在（可能是重启前的历史运行）")
        with live.lock:
            return {
                "run_id": run_id,
                "status": live.status,
                "results": dict(live.results),
                "dag": live.dag,
                "error": live.error,
                "ask": live.ask,
                "loop": live.loop,
                "events": live.events[since:],
                "next_since": len(live.events),
            }

    @app.get("/api/runs")
    def list_runs(limit: int = 30) -> dict[str, Any]:
        return {"runs": ws.runstore.runs(limit=limit)}

    @app.get("/api/runs/{run_id}/lineage")
    def get_lineage(run_id: str) -> dict[str, Any]:
        live = ws.live.get(run_id)
        if live is not None and live.lineage:
            return live.lineage
        p = ws.root / "lineage" / f"{run_id}.json"
        if not p.exists():
            raise HTTPException(404, "没有血缘记录")
        import json
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/artifacts/{kind}/{hash}")
    def get_artifact(kind: str, hash: str, limit: int = 100) -> dict[str, Any]:
        """取产物预览。DataFrame 只回前 limit 行——几十万行直接塞给前端会卡死。"""
        ref = ArtifactRef(hash=hash, kind=kind)
        if not ws.store.exists(ref):
            raise HTTPException(404, "产物不存在")
        value = ws.store.get(ref)
        if kind == "dataframe":
            head = value.head(limit)
            return {
                "kind": kind,
                "rows": int(len(value)),
                "cols": int(len(value.columns)),
                "columns": [str(c) for c in value.columns],
                "dtypes": {str(c): str(t) for c, t in value.dtypes.items()},
                "preview": head.astype(object).where(head.notna(), None).to_dict("records"),
                "truncated": len(value) > limit,
            }
        return {"kind": kind, "value": value}

    @app.post("/api/plan")
    def make_plan(req: PlanRequest) -> dict[str, Any]:
        """自然语言 → 计划书 → 表单请求体。

        **只规划，不执行。** 返回的请求体会被填进界面表单，
        用户看过、改过之后再自己点运行——这就是 Plan 这一步的人工确认。

        计划书有语义问题时返回 **200 且 ``request`` 为 null**，而不是 400：
        400 只能给一句错误，而这里要把模型到底产出了什么、哪里说不通
        一并摆给用户看，他才知道该怎么改需求描述。
        真正的 400 留给"根本没法规划"（没配模型、需求为空）。
        """
        planner = Planner(ws.llm, runstore=ws.runstore, store=ws.store)
        try:
            plan, request = planner.plan_request(req.text, seed=req.seed)
        except PlanError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001 - 模型抽风不该 500
            raise HTTPException(502, f"规划失败：{e}") from e
        return {"plan": plan.to_dict(), "request": request}

    @app.post("/api/papers")
    async def upload_paper(file: UploadFile = File(...)) -> dict[str, Any]:
        name = (file.filename or "paper.pdf").lower()
        if not name.endswith(".pdf"):
            raise HTTPException(400, "目前只支持 PDF")
        data = await file.read()
        if len(data) > 40 * 1024 * 1024:
            raise HTTPException(400, "文件超过 40 MB")
        # 按内容哈希命名：同一篇论文传两次不会占两份，
        # 而且 paper_id 稳定 → 解析节点的指纹稳定 → 缓存能命中
        import hashlib
        pid = hashlib.sha256(data).hexdigest()[:16]
        path = ws.papers / f"{pid}.pdf"
        if not path.exists():
            path.write_bytes(data)
        return {"paper_id": pid, "filename": file.filename, "size": len(data)}

    @app.get("/api/llm")
    def llm_status() -> dict[str, Any]:
        if ws.llm is None:
            return {"configured": False,
                    "hint": "设置环境变量 DATAFORGE_API_KEY 后重启即可使用论文抽取"}
        return {"configured": True, "model": getattr(ws.llm, "model", ws.llm.name),
                "usage": ws.llm.usage.to_dict()}

    @app.get("/api/defaults")
    def defaults() -> dict[str, Any]:
        from ..harness.gates import DEFAULTS
        return {"thresholds": DEFAULTS}

    # ── 静态界面 ──
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app
