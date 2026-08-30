"""质量打分。

停滞检测（§4.2）需要一个**连续**的分数：只有通过/不通过的话，
分数是阶跃的，"连续两轮提升 < ε" 就退化成 "连续两轮都没过"，
看不出「在变好但还不够」和「彻底卡住」的区别。

所以分数由两部分组成：

* **离散扣分** —— 门禁失败、泄漏发现
* **连续扣分** —— 完整性缺口、边界堆积

分数**由代码算**，绝不问模型「你觉得够好了吗」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 各类问题的扣分权重。BLOCK 级别扣得重，因为它本来就该阻断。
PENALTY = {
    "BLOCK": 40.0,
    "ASK": 15.0,
    "WARN": 8.0,
    "leak_error": 25.0,
    "leak_warning": 3.0,
}


@dataclass
class ScoreBreakdown:
    """分数的构成。要能回答「为什么是这个分」，否则调参就是盲调。"""

    score: float = 100.0
    deductions: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def deduct(self, amount: float, reason: str, **detail: Any) -> None:
        if amount <= 0:
            return
        self.score = max(0.0, self.score - amount)
        self.deductions.append({"amount": round(amount, 2), "reason": reason, **detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "deductions": self.deductions,
            "signals": self.signals,
        }


def score_gate_report(gate: dict[str, Any] | None, analyze: dict[str, Any] | None = None) -> ScoreBreakdown:
    """从门禁报告与统计结果算分。

    Args:
        gate: ``quality_gate`` 节点的产物。为 None（例如被阻断而没产出）时给 0 分。
        analyze: ``analyze`` 节点的产物，用于连续项。
    """
    b = ScoreBreakdown()

    if gate is None:
        b.score = 0.0
        b.deductions.append({"amount": 100.0, "reason": "没有产出质量报告（通常是被更早的节点阻断）"})
        return b

    results = gate.get("results", [])
    b.signals["gates_total"] = len(results)
    b.signals["gates_failed"] = sum(1 for r in results if not r.get("passed"))

    for r in results:
        if r.get("passed"):
            continue
        level = r.get("level", "WARN")
        b.deduct(PENALTY.get(level, 8.0), f"门禁 {r.get('gate')} 未通过（{level}）",
                 gate=r.get("gate"), level=level, message=r.get("message"))

    # 泄漏单独算：它是最危险的静默故障，不能和普通门禁混在一起看
    leak = next((r for r in results if r.get("gate") == "leakage"), None)
    findings = (leak or {}).get("details", {}).get("findings", [])
    n_err = sum(1 for f in findings if f.get("severity") == "error")
    n_warn = sum(1 for f in findings if f.get("severity") == "warning")
    b.signals["leak_errors"] = n_err
    b.signals["leak_warnings"] = n_warn
    if n_err:
        b.deduct(PENALTY["leak_error"] * n_err, f"检出 {n_err} 处泄漏", count=n_err)
    if n_warn:
        b.deduct(PENALTY["leak_warning"] * n_warn, f"{n_warn} 处疑似泄漏待确认", count=n_warn)

    # ── 连续项 ──
    comp = next((r for r in results if r.get("gate") == "completeness"), None)
    if comp:
        actual = float(comp.get("details", {}).get("completeness", 1.0))
        floor = float(comp.get("details", {}).get("min", 0.9))
        b.signals["completeness"] = round(actual, 4)
        gap = max(0.0, floor - actual)
        if gap > 0:
            # 缺口每差 1 个百分点扣 1.5 分，连续可导，调参时看得见进步
            b.deduct(gap * 150.0, f"完整性 {actual:.1%} 低于下限 {floor:.0%}", gap=round(gap, 4))

    rows = next((r for r in results if r.get("gate") == "row_count"), None)
    if rows:
        n = int(rows.get("details", {}).get("rows", 0))
        lo = int(rows.get("details", {}).get("min", 1))
        b.signals["rows"] = n
        if 0 < n < lo:
            # 越接近下限扣得越少，让「加了行但还不够」也能体现为进步
            b.deduct(min(20.0, (1 - n / lo) * 20.0), f"行数 {n} 未达下限 {lo}", rows=n, min=lo)

    if analyze:
        b.deduct(*_edge_piling(analyze))

    return b


def _edge_piling(analyze: dict[str, Any]) -> tuple[float, str]:
    """边界堆积：大量取值压在约束边界上，通常意味着裁剪过度而非真实分布。"""
    hist = analyze.get("histograms", {})
    worst, worst_col = 0.0, ""
    for col, h in hist.items():
        counts = h.get("counts") or []
        total = sum(counts)
        if total <= 0 or len(counts) < 4:
            continue
        edge = (counts[0] + counts[-1]) / total
        expected = 2.0 / len(counts)
        excess = max(0.0, edge - expected * 3.0)
        if excess > worst:
            worst, worst_col = excess, col
    if worst <= 0:
        return 0.0, ""
    return min(12.0, worst * 40.0), f"{worst_col} 存在边界堆积（{worst:.1%} 超出预期）"


def gate_doc_of(result: Any, store: Any) -> dict[str, Any] | None:
    """取门禁报告，**兼顾被阻断的情况**。

    门禁不通过时节点抛异常，没有产物——但明细跟着 ``NodeBlocked.details``
    进了 ``NodeResult.meta``。不从这里捞出来的话，被阻断的运行一律 0 分，
    分不出「差一点」和「差很远」，停滞检测和调参都会失灵。
    """
    if result is None:
        return None
    ref = getattr(result, "artifact", None)
    if ref is not None:
        try:
            return store.get(ref)
        except Exception:  # pragma: no cover
            pass
    meta = getattr(result, "meta", None) or {}
    if "all" in meta:
        results = meta["all"]
        return {
            "passed": all(r.get("passed") for r in results),
            "results": results,
            "audit": meta.get("audit", {}),
        }
    return None


def score_run(results: dict[str, Any], store: Any) -> ScoreBreakdown:
    """从一次运行的节点结果直接算分。

    Args:
        results: ``{node_id: NodeResult}``。
        store: 制品库，用于取回 gate / analyze 的产物。
    """
    def artifact(nid: str) -> Any:
        r = results.get(nid)
        if r is None or getattr(r, "artifact", None) is None:
            return None
        try:
            return store.get(r.artifact)
        except Exception:  # pragma: no cover - 产物被清理时不该让打分崩掉
            return None

    return score_gate_report(gate_doc_of(results.get("gate"), store), artifact("analyze"))
