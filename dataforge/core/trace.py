"""运行轨迹。

M0 的验收标准是「任意一次运行可完整重放」——重放靠的就是这里。
每个事件都记录足以重建当时状态的信息：指纹、产物哈希、耗时、错误。

轨迹是**追加写**的 JSONL，进程被杀也不会丢已写入的部分。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

EventKind = Literal[
    "run_start", "run_end",
    "node_start", "node_end", "node_cached", "node_error",
    "gate", "ask", "budget", "llm", "loop", "note",
]


@dataclass
class TraceEvent:
    kind: EventKind
    ts: float
    seq: int
    node_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trace:
    """一次运行的事件流。

    时间戳只用于展示，**不参与指纹**——否则同样的输入永远算不出同样的指纹，
    缓存和重放都会失效。
    """

    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.events: list[TraceEvent] = []
        self._seq = 0
        self._fh = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("a", encoding="utf-8")
        self.path = Path(path) if path else None

    # ── 写入 ──────────────────────────────────────────────
    def emit(self, _kind: EventKind, _node_id: str | None = None, /, **data: Any) -> TraceEvent:
        """记一个事件。

        前两个参数是**位置限定**的：事件数据里经常出现 ``kind`` / ``node_id``
        这种键（节点类型就叫 kind），若它们是普通参数就会与形参撞名，
        报 "got multiple values for argument"。位置限定从语法上杜绝这件事。
        """
        ev = TraceEvent(kind=_kind, ts=time.time(), seq=self._seq, node_id=_node_id, data=data)
        self._seq += 1
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
            self._fh.flush()  # 崩溃时也要留下已发生的事
        return ev

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── 读取 ──────────────────────────────────────────────
    def of_kind(self, *kinds: EventKind) -> list[TraceEvent]:
        return [e for e in self.events if e.kind in kinds]

    def node_events(self, node_id: str) -> list[TraceEvent]:
        return [e for e in self.events if e.node_id == node_id]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return {"run_id": self.run_id, "events": len(self.events), "by_kind": counts}

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        """从 JSONL 读回一次运行的轨迹（只读，不再追加）。"""
        p = Path(path)
        raw = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        run_id = ""
        for r in raw:
            if r["kind"] == "run_start":
                run_id = r["data"].get("run_id", "")
                break
        t = cls(run_id=run_id or p.stem)
        t.events = [TraceEvent(**r) for r in raw]
        t._seq = len(t.events)
        t.path = p
        return t

    def replay(self) -> dict[str, dict[str, Any]]:
        """从轨迹重建「每个节点算出了什么」。

        这是 M0 的验收手段：不重新执行，仅凭轨迹就能还原整次运行的
        节点指纹与产物哈希。还原不出来 = 轨迹记得不够。
        """
        out: dict[str, dict[str, Any]] = {}
        for e in self.events:
            if e.kind in ("node_end", "node_cached") and e.node_id:
                out[e.node_id] = {
                    "fingerprint": e.data.get("fingerprint"),
                    "artifact": e.data.get("artifact"),
                    "status": "CACHED" if e.kind == "node_cached" else "COMPUTED",
                }
        return out

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)
