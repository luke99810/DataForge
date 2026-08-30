"""预算与熔断。

设计文档 §4.2：循环最难的不是「怎么继续」，是「什么时候停」。
三个出口——成功 / 停滞 / 熔断——**全部由代码计算**。

其中停滞检测（plateau）是必需项不是优化项：只写「最多 5 轮」的话，
Agent 会在同一个解决不了的问题上把预算全烧完，而第 2 轮之后一点没进步，
最后还给你一份「跑满 5 轮」的努力报告。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BudgetExceeded(Exception):
    """预算耗尽。属于设计内的熔断，不是 bug。"""

    def __init__(self, what: str, used: float, limit: float) -> None:
        super().__init__(f"预算耗尽：{what} 用了 {used:g}，上限 {limit:g}")
        self.what, self.used, self.limit = what, used, limit


@dataclass
class Budget:
    """一次运行的资源上限。"""

    max_tokens: int = 200_000
    max_seconds: float = 1800.0
    max_rounds: int = 5
    max_retries_per_error: int = 3

    tokens: int = 0
    started_at: float = field(default_factory=time.time)
    rounds: int = 0
    _errors: dict[str, int] = field(default_factory=dict)

    # ── 记账 ──────────────────────────────────────────────
    def spend_tokens(self, n: int) -> None:
        self.tokens += n
        if self.tokens > self.max_tokens:
            raise BudgetExceeded("token", self.tokens, self.max_tokens)

    def tick_round(self) -> None:
        self.rounds += 1
        if self.rounds > self.max_rounds:
            raise BudgetExceeded("轮数", self.rounds, self.max_rounds)

    def record_error(self, signature: str) -> int:
        """同一个错误重复到上限就熔断——反复撞同一堵墙没有意义。"""
        n = self._errors.get(signature, 0) + 1
        self._errors[signature] = n
        if n > self.max_retries_per_error:
            raise BudgetExceeded(f"同一错误重试({signature})", n, self.max_retries_per_error)
        return n

    # ── 查询 ──────────────────────────────────────────────
    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def check_time(self) -> None:
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded("时长(秒)", round(self.elapsed, 1), self.max_seconds)

    @property
    def exhausted(self) -> bool:
        return (
            self.tokens >= self.max_tokens
            or self.rounds >= self.max_rounds
            or self.elapsed >= self.max_seconds
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "rounds": self.rounds,
            "max_rounds": self.max_rounds,
            "elapsed": round(self.elapsed, 2),
            "max_seconds": self.max_seconds,
        }


class StopReason(str, Enum):
    SUCCESS = "success"      # 门禁全绿
    PLATEAU = "plateau"      # 连续多轮没有实质进步
    EXHAUSTED = "exhausted"  # 预算耗尽
    BLOCKED = "blocked"      # 硬阻断，改参数也救不了
    NEEDS_HUMAN = "needs_human"


@dataclass
class PlateauDetector:
    """停滞检测。

    连续 ``patience`` 轮的分数提升都小于 ``epsilon``，就判定为已达当前方法上限。
    及早停下并说清楚卡在哪，比假装还在优化有用得多。
    """

    epsilon: float = 0.5
    patience: int = 2
    history: list[float] = field(default_factory=list)

    def push(self, score: float) -> bool:
        """记一轮分数，返回是否已停滞。"""
        self.history.append(float(score))
        if len(self.history) <= self.patience:
            return False
        recent = self.history[-(self.patience + 1):]
        gains = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        return all(g < self.epsilon for g in gains)

    @property
    def best(self) -> float:
        return max(self.history) if self.history else float("-inf")

    def report(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best": self.best if self.history else None,
            "epsilon": self.epsilon,
            "patience": self.patience,
        }
