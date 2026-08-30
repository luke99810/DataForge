"""血缘。

设计文档 §5.2：一份说不清来源的合成数据，在论文里是不可用的。
血缘要能回答「这一列是怎么来的」——追到节点、参数、以及人工决策。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LineageEntry:
    node_id: str
    kind: str
    fingerprint: str
    artifact: str | None
    upstream: dict[str, str] = field(default_factory=dict)   # 输入名 -> 上游 node_id
    params: dict[str, Any] = field(default_factory=dict)
    # 人工介入的记录：论文里说不清、由人拍板的地方
    decisions: list[dict[str, Any]] = field(default_factory=list)
    # 列级来源：列名 -> 说明（哪篇论文的哪一段 / 哪个规则）
    columns: dict[str, Any] = field(default_factory=dict)


class Lineage:
    """一次运行的血缘图。"""

    def __init__(self) -> None:
        self.entries: dict[str, LineageEntry] = {}

    def record(
        self,
        node_id: str,
        *,
        kind: str,
        fingerprint: str,
        artifact: str | None,
        upstream: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> LineageEntry:
        e = LineageEntry(
            node_id=node_id,
            kind=kind,
            fingerprint=fingerprint,
            artifact=artifact,
            upstream=dict(upstream or {}),
            params=dict(params or {}),
        )
        self.entries[node_id] = e
        return e

    def add_decision(self, node_id: str, *, question: str, answer: Any, by: str = "human", note: str = "") -> None:
        """记一次人工决策。

        论文单位冲突、缺失率没给——这些地方的取值不是算出来的，是人定的。
        不记下来，下游就无从判断这份数据的可信边界。
        """
        if node_id in self.entries:
            self.entries[node_id].decisions.append(
                {"question": question, "answer": answer, "by": by, "note": note}
            )

    def add_column_source(self, node_id: str, column: str, source: dict[str, Any]) -> None:
        """给某一列登记来源，例如 {'paper': 'x.pdf', 'page': 7, 'quote': '...'}。"""
        if node_id in self.entries:
            self.entries[node_id].columns[column] = source

    # ── 查询 ──────────────────────────────────────────────
    def ancestors(self, node_id: str) -> list[str]:
        """某节点的全部上游（去重，深度优先）。"""
        seen: list[str] = []
        stack = list(self.entries.get(node_id, LineageEntry("", "", "", None)).upstream.values())
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.append(n)
            e = self.entries.get(n)
            if e:
                stack.extend(e.upstream.values())
        return seen

    def descendants(self, node_id: str) -> list[str]:
        """某节点的全部下游。"""
        out: list[str] = []
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for nid, e in self.entries.items():
                if cur in e.upstream.values() and nid not in out:
                    out.append(nid)
                    frontier.append(nid)
        return out

    def explain(self, node_id: str) -> str:
        """人类可读的来源链，可直接贴进论文补充材料。"""
        e = self.entries.get(node_id)
        if not e:
            return f"未知节点 {node_id}"
        lines = [f"{node_id}  [{e.kind}]  产物 {e.artifact}"]
        for name, up in sorted(e.upstream.items()):
            lines.append(f"  ├─ 输入 {name} ← {up}")
        if e.params:
            lines.append(f"  ├─ 参数 {json.dumps(e.params, ensure_ascii=False, sort_keys=True)}")
        for d in e.decisions:
            lines.append(f"  ├─ 人工决策：{d['question']} → {d['answer']}  ({d['by']})")
        for col, src in sorted(e.columns.items()):
            lines.append(f"  ├─ 列 {col} ← {json.dumps(src, ensure_ascii=False, sort_keys=True)}")
        for a in self.ancestors(node_id):
            ae = self.entries.get(a)
            if ae:
                lines.append(f"  └─ 溯源 {a} [{ae.kind}] {ae.artifact}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            nid: {
                "node_id": e.node_id,
                "kind": e.kind,
                "fingerprint": e.fingerprint,
                "artifact": e.artifact,
                "upstream": e.upstream,
                "params": e.params,
                "decisions": e.decisions,
                "columns": e.columns,
            }
            for nid, e in self.entries.items()
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
