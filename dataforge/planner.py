"""Planner —— 自然语言 → 计划书 → 流水线请求。

设计文档 §4.1 的 Plan 一步。这里有一个刻意的取舍：

**模型产出的是「计划书」，不是图。**

让模型直接吐 DAG 很有诱惑力，但它能凭空造出不存在的节点类型、连出环、
把输入接到不存在的节点上。校验这些比校验一份扁平的计划书难得多，
出错时也很难给出能喂回模型的错误信息。

所以链路是::

    自然语言 ──► 计划书（LLM，受契约约束）
                    │
                    ├─► 语义校验（代码）—— schema 管不到的那些
                    │
                    └─► 流水线请求（代码，确定性映射）──► DAG

**模型碰不到图的结构。** 它填的是一张表单，人可以在界面上改完再跑——
这也就自然构成了 Plan 这一步的人工确认。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .llm.contracts import IDENT, PLAN_SPEC, SYSTEM_PLAN, validate_against

CODE_VERSION = "1.0.0"

_TS_TASKS = ("time_series_forecast",)


class PlanError(ValueError):
    """计划书过不了语义校验。"""


@dataclass
class Plan:
    spec: dict[str, Any]
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "issues": self.issues,
            "fixes": self.fixes,
            "cached": self.cached,
            "usage": self.usage,
        }


# ── 语义校验 ──────────────────────────────────────────────

def check_plan(spec: dict[str, Any]) -> list[str]:
    """schema 管不到的那些。

    返回问题列表（空 = 没问题）。这些是「说不通」而不是「格式错」：
    格式错要重试，说不通要么自动修，要么交给人。
    """
    issues: list[str] = []
    fields = spec.get("fields") or []
    names = [f.get("name", "") for f in fields]

    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        issues.append(f"字段名重复：{dup}")

    for f in fields:
        n = f.get("name", "")
        # str.isalnum() 在这里是陷阱：它是 Unicode 感知的，中文名会被判为合法
        if not IDENT.fullmatch(n or ""):
            issues.append(f"字段名 {n!r} 不是合法标识符（应为蛇形小写英文）")
        lo, hi = f.get("min"), f.get("max")
        if lo is not None and hi is not None and lo > hi:
            issues.append(f"{n}: min({lo}) 大于 max({hi})")
        if f.get("dtype") == "categorical" and not f.get("categories"):
            issues.append(f"{n}: categorical 必须给出 categories")

    target = spec.get("target")
    if target and target not in names:
        issues.append(f"target {target!r} 不在字段列表里：{names}")

    pipeline = spec.get("pipeline")
    task = spec.get("task_type")
    tcol = spec.get("time_column")

    if pipeline == "timeseries":
        if not tcol:
            issues.append("时序流水线必须指定 time_column")
        elif tcol not in names:
            issues.append(f"time_column {tcol!r} 不在字段列表里")
        else:
            dt = next((f for f in fields if f.get("name") == tcol), {})
            if dt.get("dtype") != "datetime":
                issues.append(f"time_column {tcol!r} 的类型是 {dt.get('dtype')!r}，应为 datetime")
    elif task in _TS_TASKS:
        # 任务是时序预测却走表格流水线 —— 会导致随机切分，虚高精度极具迷惑性
        issues.append("task_type 是时序预测，但 pipeline 不是 timeseries")

    for pair in spec.get("known_correlated") or []:
        miss = [p for p in pair if p not in names]
        if miss:
            issues.append(f"known_correlated 引用了不存在的字段：{miss}")

    return issues


def autofix(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """能确定性修好的就直接修，不必为此再问一次模型。

    只修**没有歧义**的问题。有歧义的一律留给 :func:`check_plan` 报出来。
    """
    out = json.loads(json.dumps(spec))
    fixes: list[str] = []
    fields = out.get("fields") or []
    names = [f.get("name", "") for f in fields]

    # min/max 写反了 —— 交换即可，意图很明确
    for f in fields:
        lo, hi = f.get("min"), f.get("max")
        if lo is not None and hi is not None and lo > hi:
            f["min"], f["max"] = hi, lo
            fixes.append(f"{f.get('name')}: min/max 写反了，已交换")

    # 任务是时序预测、字段里也确实有 datetime —— 补上 pipeline / time_column
    if out.get("task_type") in _TS_TASKS and out.get("pipeline") != "timeseries":
        dt = next((f for f in fields if f.get("dtype") == "datetime"), None)
        if dt:
            out["pipeline"] = "timeseries"
            out.setdefault("time_column", dt["name"])
            fixes.append(f"任务是时序预测，已改用 timeseries 流水线，时间列取 {dt['name']}")

    if out.get("pipeline") == "timeseries" and not out.get("time_column"):
        dt = next((f for f in fields if f.get("dtype") == "datetime"), None)
        if dt:
            out["time_column"] = dt["name"]
            fixes.append(f"补上 time_column = {dt['name']}")

    # target 指向不存在的字段：能唯一匹配到 role=target 就纠正，否则清空交给人
    t = out.get("target")
    if t and t not in names:
        cands = [f["name"] for f in fields if f.get("role") == "target"]
        if len(cands) == 1:
            out["target"] = cands[0]
            fixes.append(f"target {t!r} 不存在，按 role=target 纠正为 {cands[0]!r}")
        else:
            out.pop("target", None)
            fixes.append(f"target {t!r} 不存在且无法确定，已清空")

    # 引用了不存在字段的相关对，直接丢掉
    kc = out.get("known_correlated") or []
    kept = [p for p in kc if all(x in names for x in p)]
    if len(kept) != len(kc):
        out["known_correlated"] = kept
        fixes.append("丢弃了引用不存在字段的 known_correlated 项")

    return out, fixes


# ── 计划书 → 流水线请求 ──────────────────────────────────

def plan_to_request(spec: dict[str, Any], *, seed: int = 20260823) -> dict[str, Any]:
    """把计划书确定性地翻译成 ``POST /api/runs`` 的请求体。

    **这一步没有模型参与。** 计划书填的就是用户本来要手填的那张表单，
    所以翻译完可以直接进现有的 tabular / timeseries 流水线，
    不需要为 Planner 新开一条链路。
    """
    issues = check_plan(spec)
    if issues:
        raise PlanError("；".join(issues))

    fields = []
    for f in spec.get("fields", []):
        o = {"name": f["name"], "type": f["dtype"]}
        for k in ("description", "unit", "min", "max", "categories", "start", "end"):
            if f.get(k) is not None:
                o[k] = f[k]
        fields.append(o)

    return {
        "kind": spec.get("pipeline", "tabular"),
        "name": spec.get("name", "dataset"),
        "fields": fields,
        "num_records": int(spec.get("num_records", 1000)),
        "seed": seed,
        "target": spec.get("target"),
        "task_type": spec.get("task_type", "classification"),
        "time_column": spec.get("time_column"),
        "missing_rate": float(spec.get("missing_rate", 0.0)),
        "noise_ratio": float(spec.get("noise_ratio", 0.0)),
        "thresholds": {},
        "known_correlated": [list(p) for p in (spec.get("known_correlated") or [])],
        "force": [],
    }


# ── Planner ───────────────────────────────────────────────

class Planner:
    """自然语言 → 计划书。

    Args:
        llm: 模型客户端。为 None 时 :meth:`plan` 抛错——
            **不退化成关键词猜测**，那样猜出来的字段不该被当成用户的需求。
        runstore: 可选。同一段文字重复规划时直接复用，不重复花钱。
        store: 制品库，配合 runstore 使用。
    """

    def __init__(self, llm: Any, *, runstore: Any = None, store: Any = None) -> None:
        self.llm = llm
        self.runstore = runstore
        self.store = store

    def _fingerprint(self, text: str) -> str:
        model = getattr(self.llm, "model", getattr(self.llm, "name", "?"))
        payload = json.dumps(
            {"text": text.strip(), "model": model, "code_version": CODE_VERSION},
            sort_keys=True, ensure_ascii=False,
        )
        return "plan_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:26]

    def plan(self, text: str, *, max_retries: int = 3) -> Plan:
        if not text or not text.strip():
            raise PlanError("需求描述是空的")
        if self.llm is None:
            raise PlanError(
                "没有配置模型，无法规划。请设置环境变量 DATAFORGE_API_KEY 后重试。\n"
                "（不会退化成关键词猜测——猜出来的字段不该被当成你的需求。）"
            )

        fp = self._fingerprint(text)
        if self.runstore is not None and self.store is not None:
            ref = self.runstore.cache_get(fp)
            if ref is not None and self.store.exists(ref):
                spec = self.store.get(ref)
                return Plan(spec=spec, issues=check_plan(spec), cached=True)

        spec = self.llm.structured(
            system=SYSTEM_PLAN,
            user=f"用户需求：\n{text.strip()}\n\n请产出计划书。",
            schema=PLAN_SPEC,
            max_retries=max_retries,
        )
        spec, fixes = autofix(spec)
        issues = check_plan(spec)

        if not issues and self.runstore is not None and self.store is not None:
            ref = self.store.put(spec, kind="json")
            self.runstore.cache_put(fp, "planner", "PLAN", ref)

        return Plan(
            spec=spec, issues=issues, fixes=fixes,
            usage=self.llm.usage.to_dict() if hasattr(self.llm, "usage") else {},
        )

    def plan_request(self, text: str, *, seed: int = 20260823) -> tuple[Plan, dict[str, Any] | None]:
        """一步到位：规划 + 翻译成请求体。计划书有问题时请求体为 None。"""
        p = self.plan(text)
        if not p.ok:
            return p, None
        return p, plan_to_request(p.spec, seed=seed)

    # 校验函数挂在类上，方便外部按同一入口调用
    validate = staticmethod(check_plan)
