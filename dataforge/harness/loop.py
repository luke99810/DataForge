"""外层循环：Plan → Execute → Verify → Refine。

设计文档 §4：循环最难的不是「怎么继续」，是**「什么时候停」**。
所以本模块的重点全在停止判据上，而这些判据**全部由代码计算**——
绝不问模型「够好了吗」，它几乎总会说够好了。

五个出口：

======== ==========================================================
SUCCESS  门禁全绿
PLATEAU  连续若干轮质量分提升 < ε —— 已达当前方法上限
BLOCKED  改参数救不了的失败（泄漏、Schema 非法）
EXHAUSTED 预算耗尽
NEEDS_HUMAN 停在人工确认闸门
======== ==========================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..graph.dag import DAG
from .budget import Budget, BudgetExceeded, PlateauDetector, StopReason
from .refine import Proposal, RuleRefiner, assert_no_threshold_drift
from .score import ScoreBreakdown, gate_doc_of, score_run


@dataclass
class Round:
    """一轮的完整记录。要能回答「第 3 轮为什么没进步」。"""

    index: int
    params: dict[str, Any]
    run_id: str
    status: str
    score: float
    breakdown: dict[str, Any]
    proposal: dict[str, Any] | None = None
    duration: float = 0.0
    # 该轮真实的 RunReport。不进 to_dict（不可 JSON 序列化），
    # 但界面要靠它拿到最佳一轮的节点结果与产物。
    report: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "params": self.params,
            "run_id": self.run_id,
            "status": self.status,
            "score": round(self.score, 2),
            "breakdown": self.breakdown,
            "proposal": self.proposal,
            "duration": round(self.duration, 3),
        }


@dataclass
class LoopReport:
    reason: StopReason
    rounds: list[Round] = field(default_factory=list)
    best_index: int = -1
    message: str = ""
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Round | None:
        return self.rounds[self.best_index] if self.rounds and self.best_index >= 0 else None

    @property
    def ok(self) -> bool:
        return self.reason is StopReason.SUCCESS

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.rounds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "message": self.message,
            "rounds": [r.to_dict() for r in self.rounds],
            "best_index": self.best_index,
            "best_score": round(self.best.score, 2) if self.best else None,
            "scores": [round(s, 2) for s in self.scores],
            "budget": self.budget,
        }

    def __str__(self) -> str:
        lines = [f"loop [{self.reason.value}] {self.message}"]
        for r in self.rounds:
            # r.index 从 1 起，best_index 是列表下标（从 0 起）
            mark = "*" if r.index - 1 == self.best_index else " "
            tuned = ""
            if r.proposal and r.proposal.get("changes"):
                tuned = "  → " + ", ".join(
                    f"{c['param']} {c['from']}→{c['to']}" for c in r.proposal["changes"]
                )
            lines.append(f" {mark} 第{r.index}轮  分数 {r.score:6.2f}  {r.status:<12}{tuned}")
        return "\n".join(lines)


class RefineLoop:
    """把「跑 → 评 → 调」串成有明确出口的循环。

    Args:
        executor: 图执行器。
        build_dag: ``params -> DAG``。参数变了图就变，靠指纹缓存自动只重跑受影响分支。
        refiner: 调参器，默认规则式。
        plateau: 停滞检测器。
        budget: 预算。
        on_round: 每轮结束的回调，供界面实时刷新。
    """

    def __init__(
        self,
        executor: Any,
        build_dag: Callable[[dict[str, Any]], DAG],
        *,
        refiner: Any = None,
        plateau: PlateauDetector | None = None,
        budget: Budget | None = None,
        on_round: Callable[[Round], None] | None = None,
    ) -> None:
        self.executor = executor
        self.build_dag = build_dag
        self.refiner = refiner or RuleRefiner()
        self.plateau = plateau or PlateauDetector()
        self.budget = budget or Budget()
        self.on_round = on_round

    def run(
        self,
        params: dict[str, Any],
        *,
        seed: int | None = None,
        label: str = "",
        pass_score: float = 80.0,
    ) -> LoopReport:
        rounds: list[Round] = []
        cur = dict(params)
        original = dict(params)

        while True:
            # ── 预算：先问，再跑 ──
            try:
                self.budget.tick_round()
                self.budget.check_time()
            except BudgetExceeded as e:
                return self._finish(rounds, StopReason.EXHAUSTED, str(e))

            # Refine 绝不能碰阈值 —— 每轮都校验，不靠自觉
            assert_no_threshold_drift(original, cur)

            t0 = time.time()
            dag = self.build_dag(cur)
            rep = self.executor.run(dag, seed=seed, label=f"{label} r{len(rounds) + 1}")
            bd: ScoreBreakdown = score_run(rep.results, self.executor.store)

            rnd = Round(
                index=len(rounds) + 1,
                params=dict(cur),
                run_id=rep.run_id,
                status=rep.status,
                score=bd.score,
                breakdown=bd.to_dict(),
                duration=time.time() - t0,
                report=rep,
            )

            # ── 停在人工确认：不是失败，也不该继续调参 ──
            if rep.status == "needs_human":
                rounds.append(rnd)
                self._emit(rnd)
                return self._finish(
                    rounds, StopReason.NEEDS_HUMAN,
                    "停在人工确认闸门。中间产物已保留，确认后续跑只会重算受影响分支。",
                )

            # ── 成功 ──
            gate = self._artifact(rep, "gate")
            passed = bool((gate or {}).get("passed")) and bd.score >= pass_score
            if rep.ok and passed:
                rounds.append(rnd)
                self._emit(rnd)
                return self._finish(
                    rounds, StopReason.SUCCESS,
                    f"门禁全绿，质量分 {bd.score:.1f}（合格线 {pass_score:.0f}）",
                )

            # ── 提议下一轮 ──
            prop: Proposal = self.refiner.propose(
                cur, breakdown=bd.to_dict(), gate=gate, round_no=rnd.index
            )
            rnd.proposal = prop.to_dict()
            rounds.append(rnd)
            self._emit(rnd)

            if prop.give_up:
                return self._finish(rounds, StopReason.BLOCKED, prop.give_up_reason)

            # ── 停滞：分数记完再判，否则第一轮就会误判 ──
            if self.plateau.push(bd.score):
                return self._finish(
                    rounds, StopReason.PLATEAU,
                    f"连续 {self.plateau.patience} 轮提升不足 {self.plateau.epsilon}，"
                    f"已达当前方法上限（最高 {self.plateau.best:.1f}）。"
                    "继续跑只会把预算烧完而不会变好。",
                )

            if self.budget.exhausted:
                return self._finish(rounds, StopReason.EXHAUSTED, "预算耗尽")

            # 阈值不参与调参，原样带到下一轮
            cur = dict(prop.params)
            cur["thresholds"] = original.get("thresholds", {})

    # ── 内部 ──────────────────────────────────────────────
    def _artifact(self, rep: Any, nid: str) -> Any:
        # 走 gate_doc_of：被阻断时门禁明细在 NodeResult.meta 里而不是产物里
        return gate_doc_of(rep.results.get(nid), self.executor.store)

    def _emit(self, rnd: Round) -> None:
        if self.on_round is not None:
            try:
                self.on_round(rnd)
            except Exception:  # pragma: no cover - 回调不该影响循环
                pass

    def _finish(self, rounds: list[Round], reason: StopReason, message: str) -> LoopReport:
        best = max(range(len(rounds)), key=lambda i: rounds[i].score) if rounds else -1
        return LoopReport(
            reason=reason,
            rounds=rounds,
            best_index=best,
            message=message,
            budget=self.budget.snapshot(),
        )
