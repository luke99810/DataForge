"""门禁。

设计文档 §3.3：门禁是 Harness 里唯一有权**阻断流程**的组件。三级：

====== ================================ ==========================
级别   行为                              典型场景
====== ================================ ==========================
BLOCK  停止，不可自动绕过                 目标泄漏、单位冲突、Schema 非法
ASK    暂停，升级人工                     论文没给的关键参数、低置信变量
WARN   记录，允许继续                     KS 略超阈值、样本量偏小
====== ================================ ==========================

阈值可以调，但**调了必须留痕**——人在赶时间时会把阈值调低让流程过去，
那等于把唯一会变红的检查关掉了。见 :class:`GateConfig`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd

from ..graph.executor import NeedsHuman, NodeBlocked
from ..quality import leakage as lk


class Level(str, Enum):
    BLOCK = "BLOCK"
    ASK = "ASK"
    WARN = "WARN"


@dataclass
class GateResult:
    gate: str
    level: Level
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "level": self.level.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


# 默认阈值。改动会被 GateConfig 记录下来。
DEFAULTS = {
    "min_quality_score": 80.0,
    "min_completeness": 0.90,
    "max_ks": 0.10,
    "min_rows": 50,
}


@dataclass
class GateConfig:
    """门禁配置。

    任何偏离默认值的设置都进 :attr:`overrides`，最终写入运行日志与数据字典——
    让拿到这份数据的下游知道它是在什么标准下产出的。
    """

    values: dict[str, float] = field(default_factory=lambda: dict(DEFAULTS))
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def set(self, key: str, value: float, *, reason: str = "") -> "GateConfig":
        if key not in DEFAULTS:
            raise KeyError(f"未知门禁项 {key!r}，可选：{sorted(DEFAULTS)}")
        default = DEFAULTS[key]
        self.values[key] = value
        if value != default:
            relaxed = (
                value < default if key.startswith("min_") else value > default
            )
            self.overrides[key] = {
                "default": default,
                "value": value,
                "relaxed": relaxed,
                "reason": reason,
            }
        else:
            self.overrides.pop(key, None)
        return self

    def get(self, key: str) -> float:
        return self.values[key]

    @property
    def relaxed_any(self) -> bool:
        return any(o["relaxed"] for o in self.overrides.values())

    def audit(self) -> dict[str, Any]:
        """写进数据字典的那份记录。"""
        return {
            "thresholds": dict(self.values),
            "overrides": self.overrides,
            "relaxed": self.relaxed_any,
        }


# ── 具体门禁 ──────────────────────────────────────────────

Check = Callable[..., GateResult]


def gate_row_count(df: pd.DataFrame, cfg: GateConfig) -> GateResult:
    n = len(df)
    lo = int(cfg.get("min_rows"))
    return GateResult(
        "row_count", Level.BLOCK, n >= lo,
        f"行数 {n}（下限 {lo}）" if n >= lo else f"行数 {n} 低于下限 {lo}",
        {"rows": n, "min": lo},
    )


def gate_completeness(df: pd.DataFrame, cfg: GateConfig) -> GateResult:
    if df.size == 0:
        return GateResult("completeness", Level.BLOCK, False, "空数据集", {})
    ratio = float(1.0 - df.isna().to_numpy().mean())
    lo = cfg.get("min_completeness")
    return GateResult(
        "completeness", Level.WARN, ratio >= lo,
        f"完整性 {ratio:.1%}（下限 {lo:.0%}）",
        {"completeness": ratio, "min": lo},
    )


def gate_schema(df: pd.DataFrame, schema: Any) -> GateResult:
    """字段缺失属于硬阻断——后面所有分析都建立在字段存在的前提上。"""
    expected = set(schema.field_names) if hasattr(schema, "field_names") else set()
    missing = sorted(expected - set(df.columns))
    return GateResult(
        "schema", Level.BLOCK, not missing,
        "字段齐全" if not missing else f"缺少字段：{missing}",
        {"missing": missing},
    )


def gate_leakage(
    df: pd.DataFrame | None,
    *,
    target: str | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
    time_column: str | None = None,
    known_correlated: tuple = (),
) -> GateResult:
    """泄漏零容忍——设计文档 §12 的风险表里这条是 BLOCK。"""
    findings = lk.check_all(
        df, target=target, splits=splits, time_column=time_column,
        known_correlated=known_correlated,
    )
    errors = [f for f in findings if f.severity == "error"]
    return GateResult(
        "leakage", Level.BLOCK, not errors,
        "未检出泄漏" if not errors else f"检出 {len(errors)} 处泄漏",
        {"findings": [f.to_dict() for f in findings]},
    )


def gate_unit_conflict(conflicts: list[dict[str, Any]]) -> GateResult:
    """单位冲突不允许自动选一个继续——升级人工。"""
    return GateResult(
        "unit_conflict", Level.ASK, not conflicts,
        "无单位冲突" if not conflicts else f"{len(conflicts)} 处单位冲突需人工确认",
        {"conflicts": conflicts},
    )


# ── 汇总 ──────────────────────────────────────────────────

@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)
    config: GateConfig = field(default_factory=GateConfig)

    def add(self, r: GateResult) -> "GateReport":
        self.results.append(r)
        return self

    @property
    def blocked(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and r.level is Level.BLOCK]

    @property
    def asks(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and r.level is Level.ASK]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and r.level is Level.WARN]

    @property
    def passed(self) -> bool:
        return not self.blocked and not self.asks

    def raise_if_needed(self) -> None:
        """把门禁结果转成执行器认识的异常。

        BLOCK 优先于 ASK：既然已经硬失败了，就不该再去问人。
        """
        if self.blocked:
            first = self.blocked[0]
            raise NodeBlocked(
                f"[{first.gate}] {first.message}",
                gate=first.gate,
                details={"all": [r.to_dict() for r in self.results]},
            )
        if self.asks:
            first = self.asks[0]
            raise NeedsHuman(
                f"[{first.gate}] {first.message}",
                context={"all": [r.to_dict() for r in self.results]},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "audit": self.config.audit(),
        }
