"""泄漏检测。

设计文档 §8 把这个列为差异化点：泄漏是合成数据最典型的**静默故障**——
指标好得离谱，上线全崩，而质量分可能只掉几分。

三类：

1. 目标泄漏 —— 某特征与标签相关性异常高
2. 切分重叠 —— train / test 存在相同或近重复样本
3. 时间穿越 —— 时序任务用了随机切分，或特征窗口跨过切分点

关于第 1 类有个必须处理的误报：**物理规律导致的高相关不是泄漏**。
交通流里 occupancy 与 volume 天然强相关，报成泄漏就是噪音。
所以提供 ``known_correlated`` 白名单，并在结论里说明理由。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class LeakageFinding:
    kind: str          # target | overlap | temporal
    severity: str      # error | warning | info
    message: str
    columns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "columns": self.columns,
            "details": self.details,
        }


# 相关性到这个程度基本可以断定是泄漏而非真实信号
TARGET_CORR_ERROR = 0.98
TARGET_CORR_WARN = 0.90


def check_target_leakage(
    df: pd.DataFrame,
    target: str,
    *,
    known_correlated: Iterable[Sequence[str]] = (),
    error_threshold: float = TARGET_CORR_ERROR,
    warn_threshold: float = TARGET_CORR_WARN,
) -> list[LeakageFinding]:
    """检查特征与标签的相关性是否高到不合理。

    Args:
        known_correlated: 已知的合理强相关对，如 [("occupancy", "volume")]。
            命中白名单的只记 info，不报警。
    """
    out: list[LeakageFinding] = []
    if target not in df.columns:
        return out

    white = {frozenset(pair) for pair in known_correlated}
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        # 分类目标：用「某特征能否几乎唯一确定标签」来判断
        return _categorical_target_leakage(df, target, white)

    for col in df.columns:
        if col == target or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        pair = df[[col, target]].dropna()
        if len(pair) < 10 or pair[col].nunique() < 2:
            continue
        r = float(abs(pair[col].corr(pair[target])))
        if np.isnan(r):
            continue

        whitelisted = frozenset((col, target)) in white
        if r >= error_threshold and not whitelisted:
            out.append(LeakageFinding(
                "target", "error",
                f"{col} 与目标 {target} 相关性 {r:.3f}，几乎可以直接推出标签",
                [col, target], {"corr": r, "threshold": error_threshold},
            ))
        elif r >= warn_threshold:
            sev = "info" if whitelisted else "warning"
            reason = "已列入已知合理强相关，判定为物理/业务规律" if whitelisted else "需人工确认是否为泄漏"
            out.append(LeakageFinding(
                "target", sev,
                f"{col} 与目标 {target} 相关性 {r:.3f}——{reason}",
                [col, target], {"corr": r, "whitelisted": whitelisted},
            ))
    return out


def _categorical_target_leakage(
    df: pd.DataFrame, target: str, white: set[frozenset]
) -> list[LeakageFinding]:
    """分类目标：如果某列的每个取值都对应唯一标签，那它就是标签的马甲。"""
    out: list[LeakageFinding] = []
    for col in df.columns:
        if col == target:
            continue
        sub = df[[col, target]].dropna()
        if len(sub) < 10:
            continue
        n_groups = sub[col].nunique()
        if n_groups < 2 or n_groups > max(50, len(sub) // 10):
            continue  # 取值太多的列（近似唯一键）另行判断
        purity = sub.groupby(col, observed=True)[target].nunique()
        pure_ratio = float((purity == 1).mean())
        if pure_ratio >= 0.99 and frozenset((col, target)) not in white:
            out.append(LeakageFinding(
                "target", "error",
                f"{col} 的每个取值都唯一对应一个 {target}，是标签的另一种编码",
                [col, target], {"pure_ratio": pure_ratio, "groups": int(n_groups)},
            ))
    return out


def check_split_overlap(
    splits: dict[str, pd.DataFrame],
    *,
    subset: Sequence[str] | None = None,
) -> list[LeakageFinding]:
    """检查各 split 之间有没有重复行。"""
    out: list[LeakageFinding] = []
    names = list(splits)
    keyed: dict[str, set[int]] = {}
    for n, d in splits.items():
        use = d[list(subset)] if subset else d
        keyed[n] = set(pd.util.hash_pandas_object(use, index=False).tolist())

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = keyed[a] & keyed[b]
            if inter:
                ratio = len(inter) / max(1, min(len(keyed[a]), len(keyed[b])))
                out.append(LeakageFinding(
                    "overlap", "error",
                    f"{a} 与 {b} 有 {len(inter)} 行完全重复（占较小集合的 {ratio:.1%}）",
                    list(subset) if subset else [],
                    {"a": a, "b": b, "count": len(inter), "ratio": ratio},
                ))
    return out


def check_temporal_leakage(
    splits: dict[str, pd.DataFrame],
    time_column: str,
    *,
    order: Sequence[str] = ("train", "val", "test"),
) -> list[LeakageFinding]:
    """时序任务：切分必须严格按时间前后，不能有交叉。

    随机切分在时序任务上几乎总是错的，而它产生的虚高精度极具迷惑性——
    所以这里判 error，不是 warning。
    """
    out: list[LeakageFinding] = []
    present = [s for s in order if s in splits and time_column in splits[s].columns]
    if len(present) < 2:
        return out

    spans: dict[str, tuple[Any, Any]] = {}
    for s in present:
        col = splits[s][time_column].dropna()
        if col.empty:
            continue
        spans[s] = (col.min(), col.max())

    keys = [s for s in present if s in spans]
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        a_max, b_min = spans[a][1], spans[b][0]
        if b_min < a_max:
            out.append(LeakageFinding(
                "temporal", "error",
                f"{b} 的起点 {b_min} 早于 {a} 的终点 {a_max}——时间上有交叉，"
                f"疑似使用了随机切分。时序任务必须按时间顺序切分。",
                [time_column],
                {"earlier": a, "later": b, "a_max": str(a_max), "b_min": str(b_min)},
            ))
    return out


def check_all(
    df: pd.DataFrame | None = None,
    *,
    target: str | None = None,
    splits: dict[str, pd.DataFrame] | None = None,
    time_column: str | None = None,
    known_correlated: Iterable[Sequence[str]] = (),
    subset: Sequence[str] | None = None,
) -> list[LeakageFinding]:
    """一次跑完三类检查。"""
    findings: list[LeakageFinding] = []
    if df is not None and target:
        findings += check_target_leakage(df, target, known_correlated=known_correlated)
    if splits:
        findings += check_split_overlap(splits, subset=subset)
        if time_column:
            findings += check_temporal_leakage(splits, time_column)
    return findings


def has_error(findings: Iterable[LeakageFinding]) -> bool:
    return any(f.severity == "error" for f in findings)
