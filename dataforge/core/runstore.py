"""运行记录与指纹缓存索引（SQLite）。

两张表：

``cache``  指纹 → 产物。**跨运行**复用的关键，也是「改一个字段只重跑受影响
           分支」这条验收标准的实现基础。
``runs``   每次运行的元信息，供界面列出历史。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRef

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    fingerprint   TEXT PRIMARY KEY,
    node_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    artifact_meta TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    hits          INTEGER NOT NULL DEFAULT 0,
    source_hash   TEXT
);
CREATE INDEX IF NOT EXISTS idx_cache_node ON cache(node_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    label      TEXT,
    status     TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL,
    nodes      INTEGER NOT NULL DEFAULT 0,
    cached     INTEGER NOT NULL DEFAULT 0,
    trace_path TEXT,
    summary    TEXT NOT NULL DEFAULT '{}'
);
"""


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 本地服务的同步端点跑在线程池里，连接会被不同线程碰到。
        # 放开 sqlite 的线程检查，并用一把锁把所有访问串起来——
        # 单进程本地库，锁的开销可以忽略，比每次开新连接省事得多。
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # ── 缓存 ──────────────────────────────────────────────
    def cache_get(self, fingerprint: str) -> ArtifactRef | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT artifact_hash, artifact_kind, artifact_meta FROM cache WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE cache SET hits = hits + 1 WHERE fingerprint = ?", (fingerprint,)
            )
            self.conn.commit()
            return ArtifactRef(
                hash=row["artifact_hash"],
                kind=row["artifact_kind"],
                meta=json.loads(row["artifact_meta"]),
            )

    def cache_put(
        self, fingerprint: str, node_id: str, kind: str, ref: ArtifactRef,
        source_hash: str | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache"
                " (fingerprint, node_id, kind, artifact_hash, artifact_kind, artifact_meta,"
                "  created_at, hits, source_hash)"
                " VALUES (?,?,?,?,?,?,?,COALESCE((SELECT hits FROM cache WHERE fingerprint=?),0),?)",
                (
                    fingerprint,
                    node_id,
                    kind,
                    ref.hash,
                    ref.kind,
                    json.dumps(ref.meta, ensure_ascii=False, sort_keys=True),
                    time.time(),
                    fingerprint,
                    source_hash,
                ),
            )
            self.conn.commit()

    def cache_source_hash(self, fingerprint: str) -> str | None:
        """取缓存条目当时记录的实现哈希，用于漂移检查。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT source_hash FROM cache WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            return row["source_hash"] if row else None

    def cache_clear(self, node_id: str | None = None) -> int:
        with self._lock:
            cur = (
                self.conn.execute("DELETE FROM cache WHERE node_id = ?", (node_id,))
                if node_id
                else self.conn.execute("DELETE FROM cache")
            )
            self.conn.commit()
            return cur.rowcount

    def cache_stats(self) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(hits),0) h FROM cache"
            ).fetchone()
            return {"entries": row["n"], "total_hits": row["h"]}

    # ── 运行 ──────────────────────────────────────────────
    def run_start(self, run_id: str, label: str = "", trace_path: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, label, status, started_at, trace_path)"
                " VALUES (?,?,?,?,?)",
                (run_id, label, "running", time.time(), trace_path),
            )
            self.conn.commit()

    def run_end(
        self, run_id: str, status: str, *, nodes: int = 0, cached: int = 0, summary: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE runs SET status=?, ended_at=?, nodes=?, cached=?, summary=? WHERE run_id=?",
                (
                    status,
                    time.time(),
                    nodes,
                    cached,
                    json.dumps(summary or {}, ensure_ascii=False, sort_keys=True),
                    run_id,
                ),
            )
            self.conn.commit()

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
