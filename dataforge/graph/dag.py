"""DAG 构建、校验与拓扑排序。"""

from __future__ import annotations

from typing import Iterable, Iterator

from .node import NodeSpec


class CycleError(ValueError):
    """图里有环。"""


class DAG:
    def __init__(self, nodes: Iterable[NodeSpec] | None = None) -> None:
        self._nodes: dict[str, NodeSpec] = {}
        for n in nodes or []:
            self.add(n)

    # ── 构建 ──────────────────────────────────────────────
    def add(self, spec: NodeSpec) -> "DAG":
        if spec.node_id in self._nodes:
            raise ValueError(f"节点 {spec.node_id} 重复")
        self._nodes[spec.node_id] = spec
        return self

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __getitem__(self, node_id: str) -> NodeSpec:
        return self._nodes[node_id]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[NodeSpec]:
        return iter(self._nodes.values())

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes)

    # ── 校验 ──────────────────────────────────────────────
    def validate(self) -> None:
        """悬空输入与环都在这里拦下，不留到执行期才炸。"""
        for spec in self._nodes.values():
            for name, up in spec.inputs.items():
                if up not in self._nodes:
                    raise ValueError(
                        f"节点 {spec.node_id} 的输入 {name!r} 指向不存在的节点 {up!r}"
                    )
        self.topo_order()  # 有环会抛

    # ── 拓扑 ──────────────────────────────────────────────
    def topo_order(self) -> list[str]:
        """确定性拓扑序。

        同层节点按 node_id 排序 —— 顺序不稳定的话，轨迹就没法逐行比对，
        重放验证会假阳性。
        """
        indeg = {nid: 0 for nid in self._nodes}
        children: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for nid, spec in self._nodes.items():
            for up in set(spec.inputs.values()):
                indeg[nid] += 1
                children[up].append(nid)

        ready = sorted([nid for nid, d in indeg.items() if d == 0])
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            newly: list[str] = []
            for ch in children[cur]:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    newly.append(ch)
            if newly:
                ready = sorted(ready + newly)

        if len(order) != len(self._nodes):
            stuck = sorted(set(self._nodes) - set(order))
            raise CycleError(f"图中存在环，涉及节点：{stuck}")
        return order

    # ── 关系 ──────────────────────────────────────────────
    def parents(self, node_id: str) -> list[str]:
        return sorted(set(self._nodes[node_id].inputs.values()))

    def children(self, node_id: str) -> list[str]:
        return sorted(nid for nid, s in self._nodes.items() if node_id in s.inputs.values())

    def descendants(self, node_id: str) -> list[str]:
        """全部下游。失效传播就是按这个集合走的。"""
        out: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for ch in self.children(cur):
                if ch not in out:
                    out.add(ch)
                    frontier.append(ch)
        return sorted(out)

    def ancestors(self, node_id: str) -> list[str]:
        out: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for p in self.parents(cur):
                if p not in out:
                    out.add(p)
                    frontier.append(p)
        return sorted(out)

    def subgraph_to(self, targets: Iterable[str]) -> list[str]:
        """跑出 targets 所需的最小节点集合（拓扑序）。"""
        need: set[str] = set()
        for t in targets:
            need.add(t)
            need.update(self.ancestors(t))
        return [n for n in self.topo_order() if n in need]

    def to_dict(self) -> dict:
        """给界面画图用。"""
        return {
            "nodes": [
                {
                    "id": s.node_id,
                    "kind": s.kind,
                    "description": s.description,
                    "params": s.params,
                    "code_version": s.code_version,
                    "cacheable": s.cacheable,
                }
                for s in self._nodes.values()
            ],
            "edges": [
                {"from": up, "to": s.node_id, "name": name}
                for s in self._nodes.values()
                for name, up in s.inputs.items()
            ],
        }
