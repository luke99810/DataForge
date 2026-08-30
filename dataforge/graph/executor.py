"""图执行器 —— 指纹缓存、失效传播、轨迹与血缘。

两条验收标准（设计文档 §11）由这里保证：

1. **任意一次运行可完整重放** —— 每个节点的指纹与产物哈希都进轨迹。
2. **改一个字段只重跑受影响分支** —— 指纹随上游变化，未受影响的节点命中缓存。

失效传播是**自动**的：上游产物变 → 下游指纹的 ``upstream`` 部分变 →
下游缓存不命中。不需要显式的失效算法，但需要显式的**报告**，
否则用户看不出哪些被重算了。
"""

from __future__ import annotations

import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.artifacts import ArtifactRef, ArtifactStore
from ..core.fingerprint import fingerprint, source_hash
from ..core.lineage import Lineage
from ..core.runstore import RunStore
from ..core.trace import Trace
from .dag import DAG
from .node import NodeContext, NodeSpec


class ExecStatus(str, Enum):
    COMPUTED = "COMPUTED"   # 真算了
    CACHED = "CACHED"       # 命中缓存
    SKIPPED = "SKIPPED"     # 上游失败被跳过
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"     # 被门禁阻断


@dataclass
class NodeResult:
    node_id: str
    kind: str
    status: ExecStatus
    fingerprint: str
    artifact: ArtifactRef | None = None
    duration: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class NodeBlocked(Exception):
    """节点抛这个表示「门禁不通过，必须停」。

    与普通异常区分开：普通异常是 bug，这个是**设计内的阻断**。
    """

    def __init__(self, message: str, *, gate: str = "", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.gate = gate
        self.details = details or {}


class NeedsHuman(Exception):
    """节点抛这个表示「这里我不替你猜，要人来定」。

    对应设计文档 §3.3 的 ASK 级门禁。执行器会保留全部中间产物后停下。
    """

    def __init__(self, question: str, *, options: list[Any] | None = None, context: dict[str, Any] | None = None) -> None:
        super().__init__(question)
        self.question = question
        self.options = options or []
        self.context = context or {}


@dataclass
class RunReport:
    run_id: str
    status: str
    results: dict[str, NodeResult]
    order: list[str]
    trace: Trace
    lineage: Lineage
    started_at: float
    ended_at: float

    @property
    def computed(self) -> list[str]:
        return [n for n, r in self.results.items() if r.status is ExecStatus.COMPUTED]

    @property
    def cached(self) -> list[str]:
        return [n for n, r in self.results.items() if r.status is ExecStatus.CACHED]

    @property
    def failed(self) -> list[str]:
        return [n for n, r in self.results.items() if r.status in (ExecStatus.FAILED, ExecStatus.BLOCKED)]

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "nodes": len(self.results),
            "computed": len(self.computed),
            "cached": len(self.cached),
            "failed": len(self.failed),
            "duration": round(self.ended_at - self.started_at, 3),
        }

    def __str__(self) -> str:
        lines = [f"run {self.run_id}  [{self.status}]"]
        for nid in self.order:
            r = self.results.get(nid)
            if r is None:
                continue
            mark = {
                ExecStatus.COMPUTED: "*",
                ExecStatus.CACHED: "=",
                ExecStatus.SKIPPED: "-",
                ExecStatus.FAILED: "!",
                ExecStatus.BLOCKED: "x",
            }[r.status]
            extra = f"  {r.error}" if r.error else ""
            lines.append(f"  {mark} {nid:<24} {r.status.value:<9} {r.duration:6.3f}s  {r.artifact}{extra}")
        return "\n".join(lines)


class Executor:
    """按拓扑序执行 DAG。

    Args:
        store: 制品库。
        runstore: 指纹缓存与运行记录。
        workdir: 轨迹等文件的落地目录。
        on_event: 每个节点结束后的回调，供界面实时刷新。
    """

    def __init__(
        self,
        store: ArtifactStore,
        runstore: RunStore,
        *,
        workdir: str | Path | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        strict_drift: bool = False,
        llm: Any = None,
        fetcher: Any = None,
    ) -> None:
        self.store = store
        self.runstore = runstore
        self.workdir = Path(workdir) if workdir else Path(store.root).parent
        self.on_event = on_event
        # True 时「改了实现却没升 code_version」直接阻断而不是警告。
        # CI 里应当开启：这个故障静默起来后果比停下来严重得多。
        self.strict_drift = strict_drift
        # 唯一持有 LLM 客户端与取数器的地方，按需注入给节点。
        # 它们都不进 params，因此不会进指纹，也不会把密钥写进轨迹。
        self.llm = llm
        self.fetcher = fetcher

    def run(
        self,
        dag: DAG,
        *,
        run_id: str | None = None,
        seed: int | None = None,
        targets: Iterable[str] | None = None,
        force: Iterable[str] | None = None,
        label: str = "",
    ) -> RunReport:
        """执行图。

        Args:
            targets: 只跑这些节点及其上游。None 则跑全图。
            force: 强制重算的节点（连同其下游一起失效）。
        """
        dag.validate()
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        order = dag.subgraph_to(targets) if targets else dag.topo_order()

        # force 的下游也必须重算，否则会拿着旧上游的缓存产物往下走
        forced: set[str] = set()
        for f in force or []:
            forced.add(f)
            forced.update(dag.descendants(f))

        trace_path = self.workdir / "traces" / f"{run_id}.jsonl"
        trace = Trace(run_id, trace_path)
        lineage = Lineage()
        results: dict[str, NodeResult] = {}
        started = time.time()

        self.runstore.run_start(run_id, label=label, trace_path=str(trace_path))
        trace.emit(
            "run_start",
            run_id=run_id,
            seed=seed,
            nodes=len(order),
            order=order,
            forced=sorted(forced),
        )

        status = "success"
        try:
            for nid in order:
                spec = dag[nid]
                r = self._run_node(spec, results, trace, lineage, run_id, seed, forced)
                results[nid] = r
                if r.status in (ExecStatus.FAILED, ExecStatus.BLOCKED):
                    status = "blocked" if r.status is ExecStatus.BLOCKED else "failed"
                    # 下游全部跳过，但保留已产出的制品
                    for down in dag.descendants(nid):
                        if down not in results:
                            results[down] = NodeResult(
                                node_id=down,
                                kind=dag[down].kind,
                                status=ExecStatus.SKIPPED,
                                fingerprint="",
                                error=f"上游 {nid} {r.status.value}",
                            )
                    break
        except NeedsHuman as e:
            status = "needs_human"
            trace.emit("ask", question=e.question, options=e.options, context=e.context)
        finally:
            ended = time.time()
            trace.emit(
                "run_end",
                run_id=run_id,
                status=status,
                computed=[n for n, r in results.items() if r.status is ExecStatus.COMPUTED],
                cached=[n for n, r in results.items() if r.status is ExecStatus.CACHED],
            )
            self.runstore.run_end(
                run_id,
                status,
                nodes=len(results),
                cached=sum(1 for r in results.values() if r.status is ExecStatus.CACHED),
                summary={"order": order},
            )
            lineage.save(self.workdir / "lineage" / f"{run_id}.json")
            trace.close()

        return RunReport(
            run_id=run_id,
            status=status,
            results=results,
            order=order,
            trace=trace,
            lineage=lineage,
            started_at=started,
            ended_at=ended,
        )

    # ── 单节点 ────────────────────────────────────────────
    def _run_node(
        self,
        spec: NodeSpec,
        results: dict[str, NodeResult],
        trace: Trace,
        lineage: Lineage,
        run_id: str,
        seed: int | None,
        forced: set[str],
    ) -> NodeResult:
        # 上游产物哈希 —— 指纹的核心组成，失效靠它自动向下传播
        upstream_hashes: dict[str, str] = {}
        for name, up in spec.inputs.items():
            up_res = results.get(up)
            if up_res is None or up_res.artifact is None:
                return NodeResult(
                    node_id=spec.node_id,
                    kind=spec.kind,
                    status=ExecStatus.SKIPPED,
                    fingerprint="",
                    error=f"上游 {up} 无产物",
                )
            upstream_hashes[name] = up_res.artifact.hash

        fp = fingerprint(
            node_id=spec.node_id,
            kind=spec.kind,
            upstream=upstream_hashes,
            params=spec.params,
            code_version=spec.code_version,
            seed=seed,
        )

        lineage.record(
            spec.node_id,
            kind=spec.kind,
            fingerprint=fp,
            artifact=None,
            upstream=spec.inputs,
            params=spec.params,
        )

        # ── 缓存 ──
        use_cache = spec.cacheable and spec.node_id not in forced
        cur_src = source_hash(spec.fn)
        if use_cache:
            ref = self.runstore.cache_get(fp)
            if ref is not None and self.store.exists(ref):
                # 实现漂移检查：指纹一样但源码变了 = 改了代码却没升 code_version。
                # 这正是 §5.2 那个「静默返回旧数据」的故障，必须让它可见。
                old_src = self.runstore.cache_source_hash(fp)
                if cur_src and old_src and cur_src != old_src:
                    msg = (
                        f"节点 {spec.node_id} 的实现变了，但 code_version 仍是 "
                        f"{spec.code_version!r}——缓存会返回旧算法的结果。请升版本号。"
                    )
                    trace.emit("note", spec.node_id, drift=True, message=msg,
                               old_source=old_src, new_source=cur_src)
                    if self.strict_drift:
                        # 这里在 spec.fn 的 try/except 之外，抛异常会穿透执行器，
                        # 所以直接构造 BLOCKED 结果返回
                        trace.emit("node_error", spec.node_id, fingerprint=fp,
                                   error=msg, blocked=True)
                        return NodeResult(
                            node_id=spec.node_id,
                            kind=spec.kind,
                            status=ExecStatus.BLOCKED,
                            fingerprint=fp,
                            error=msg,
                            meta={"gate": "code_drift"},
                        )
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                lineage.entries[spec.node_id].artifact = ref.hash
                self._emit(
                    trace, "node_cached", spec.node_id,
                    node_kind=spec.kind,
                    fingerprint=fp,
                    artifact=ref.to_dict(),
                )

                return NodeResult(
                    node_id=spec.node_id,
                    kind=spec.kind,
                    status=ExecStatus.CACHED,
                    fingerprint=fp,
                    artifact=ref,
                )

        # ── 真算 ──
        self._emit(trace, "node_start", spec.node_id, node_kind=spec.kind, fingerprint=fp)
        t0 = time.time()

        inputs = {
            name: self.store.get(results[up].artifact)  # type: ignore[arg-type]
            for name, up in spec.inputs.items()
        }
        ctx = NodeContext(
            node_id=spec.node_id, run_id=run_id, seed=seed,
            workdir=self.workdir, llm=self.llm, fetcher=self.fetcher,
        )

        try:
            value = spec.fn(inputs, dict(spec.params), ctx)
        except NeedsHuman:
            raise
        except NodeBlocked as e:
            dur = time.time() - t0
            self._emit(trace, "gate", spec.node_id, blocked=True, gate=e.gate,
                       message=str(e), details=e.details)
            self._emit(trace, "node_error", spec.node_id, fingerprint=fp,
                       error=str(e), blocked=True)
            return NodeResult(
                node_id=spec.node_id,
                kind=spec.kind,
                status=ExecStatus.BLOCKED,
                fingerprint=fp,
                duration=dur,
                error=str(e),
                meta=e.details,
            )
        except Exception as e:  # noqa: BLE001 - 节点异常要变成结果，不能炸掉整个执行器
            dur = time.time() - t0
            self._emit(
                trace, "node_error", spec.node_id,
                fingerprint=fp,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=6),
            )
            return NodeResult(
                node_id=spec.node_id,
                kind=spec.kind,
                status=ExecStatus.FAILED,
                fingerprint=fp,
                duration=dur,
                error=f"{type(e).__name__}: {e}",
            )

        dur = time.time() - t0
        ref = self.store.put(value, kind=spec.artifact_kind)
        if spec.cacheable:
            self.runstore.cache_put(fp, spec.node_id, spec.kind, ref, source_hash=cur_src)

        # 节点写进 ctx 的血缘信息落库
        entry = lineage.entries[spec.node_id]
        entry.artifact = ref.hash
        for col, src in ctx.lineage_notes.get("columns", {}).items():
            entry.columns[col] = src
        for d in ctx.lineage_notes.get("decisions", []):
            entry.decisions.append(d)
        for n in ctx.trace_notes:
            trace.emit("note", spec.node_id, **n)

        self._emit(
            trace, "node_end", spec.node_id,
            node_kind=spec.kind,
            fingerprint=fp,
            artifact=ref.to_dict(),
            duration=round(dur, 4),
        )
        return NodeResult(
            node_id=spec.node_id,
            kind=spec.kind,
            status=ExecStatus.COMPUTED,
            fingerprint=fp,
            artifact=ref,
            duration=dur,
        )

    def _emit(self, trace: Trace, _kind: str, _node_id: str | None = None, /, **data: Any) -> None:
        """写轨迹并同步通知界面。

        两边用同一份数据，避免「界面看到的」与「轨迹记录的」不一致。
        注意 data 里的节点类型字段叫 ``node_kind`` 而不是 ``kind``——
        后者会与事件类型撞名并把它覆盖掉。
        """
        ev = trace.emit(_kind, _node_id, **data)
        self._notify(ev.kind, {"node_id": ev.node_id, "seq": ev.seq, **ev.data})

    def _notify(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, data)
            except Exception:  # pragma: no cover - 回调不该影响执行
                pass
