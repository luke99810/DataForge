"""Refine —— 提出下一轮的参数调整。

设计文档 §4.2 划的红线：

* **可以改**：分布参数、噪声强度、样本量、采样策略、缺失注入率
* **不可以改**：已经过人工确认的 Schema；**门禁阈值**；跳过 BLOCK 级门禁

  如果允许 Refine 调阈值，循环一定会「收敛」——因为它会把标准降到自己
  够得着为止。那不是收敛，是自欺。这里从代码上堵死：
  :class:`RuleRefiner` 根本不接触 thresholds，且 :func:`assert_no_threshold_drift`
  会在每轮校验。

另外要诚实的一点：**Refine 只有很少几个可用的杠杆**。
有些失败（比如目标泄漏、Schema 非法）改参数是修不好的——
那种情况要明确停在 ``BLOCKED``，而不是白白把预算烧完。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Refine 允许触碰的参数。不在这个白名单里的一律不改。
TUNABLE = ("num_records", "missing_rate", "noise_ratio")

# 改参数救不了的门禁——出现这些就该停，不该继续试
UNFIXABLE_GATES = ("leakage", "schema", "unit_conflict", "code_drift")


@dataclass
class Proposal:
    """一次调参提议。"""

    params: dict[str, Any]
    changes: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    give_up: bool = False          # 这类问题调参救不了
    give_up_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "changes": self.changes,
            "rationale": self.rationale,
            "give_up": self.give_up,
            "give_up_reason": self.give_up_reason,
        }


class RuleRefiner:
    """规则式调参。

    为什么先用规则而不是模型：常见的几种失败（行数不够、缺失率过高）
    与修法是一一对应的，规则**确定、可测、免费**。
    模型留给规则覆盖不到的情况（:class:`LLMRefiner`）。
    """

    name = "rule"

    def __init__(self, *, max_records: int = 500_000) -> None:
        self.max_records = max_records

    def propose(
        self,
        params: dict[str, Any],
        *,
        breakdown: dict[str, Any],
        gate: dict[str, Any] | None,
        round_no: int = 1,
    ) -> Proposal:
        new = dict(params)
        changes: list[dict[str, Any]] = []
        results = (gate or {}).get("results", [])
        failed = {r["gate"]: r for r in results if not r.get("passed")}

        # ── 先判断「这问题调参救不了」 ──
        for g in UNFIXABLE_GATES:
            r = failed.get(g)
            if r is not None:
                return Proposal(
                    params=new, give_up=True,
                    give_up_reason=(
                        f"门禁 {g} 未通过：{r.get('message', '')}。"
                        "这类问题改参数修不好——泄漏要改切分或特征，"
                        "Schema 非法要回到变量确认。继续试只是浪费预算。"
                    ),
                )

        # ── 行数不够 ──
        rows = failed.get("row_count")
        if rows:
            need = int(rows.get("details", {}).get("min", 50))
            cur = int(new.get("num_records", 0))
            # 一次到位并留 20% 余量，别一点一点加浪费轮数
            target = min(self.max_records, max(need * 2, int(need * 1.2), cur * 2, need))
            if target > cur:
                new["num_records"] = target
                changes.append({"param": "num_records", "from": cur, "to": target,
                                "why": f"行数 {cur} 低于门禁下限 {need}"})

        # ── 完整性不够（缺失注入过多） ──
        comp = failed.get("completeness")
        if comp:
            actual = float(comp.get("details", {}).get("completeness", 1.0))
            floor = float(comp.get("details", {}).get("min", 0.9))
            cur = float(new.get("missing_rate", 0.0))
            if cur > 0:
                # 直接按缺口折算，而不是固定步长
                target = max(0.0, min(cur, (1.0 - floor) * 0.8))
                if target < cur - 1e-9:
                    new["missing_rate"] = round(target, 4)
                    changes.append({"param": "missing_rate", "from": cur, "to": round(target, 4),
                                    "why": f"完整性 {actual:.1%} 低于下限 {floor:.0%}"})
            else:
                return Proposal(
                    params=new, give_up=True,
                    give_up_reason=(
                        f"完整性 {actual:.1%} 不达标，但 missing_rate 已经是 0——"
                        "缺失不是注入进来的，调参没有着力点。"
                    ),
                )

        # ── 边界堆积：降噪声 ──
        if any("边界堆积" in d.get("reason", "") for d in breakdown.get("deductions", [])):
            cur = float(new.get("noise_ratio", 0.0))
            if cur > 0.01:
                target = round(cur * 0.5, 4)
                new["noise_ratio"] = target
                changes.append({"param": "noise_ratio", "from": cur, "to": target,
                                "why": "取值大量压在约束边界，通常是噪声过大后被裁剪"})

        if not changes:
            return Proposal(
                params=new, give_up=True,
                give_up_reason=(
                    "没有可调的参数了。当前失败项不在 Refine 的杠杆范围内"
                    f"（可调：{', '.join(TUNABLE)}）。"
                ),
            )

        return Proposal(
            params=new, changes=changes,
            rationale="；".join(c["why"] for c in changes),
        )


class LLMRefiner:
    """模型式调参 —— 规则给不出提议时的兜底。

    模型只被允许**在白名单参数上提议数值**，返回的提议仍要过
    :func:`sanitize`：越界的、碰阈值的、改 Schema 的一律丢弃。
    **提议归模型，采纳与否归代码。**
    """

    name = "llm"

    SYSTEM = (
        "你在调一个合成数据生成器的参数。只能调整给定的参数，"
        "不得建议修改门禁阈值或数据字段定义。只输出 JSON。"
    )
    SCHEMA = {
        "type": "object",
        "required": ["params", "rationale"],
        "properties": {
            "params": {"type": "object"},
            "rationale": {"type": "string"},
            "give_up": {"type": "boolean"},
        },
    }

    def __init__(self, llm: Any, fallback: RuleRefiner | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RuleRefiner()

    def propose(
        self,
        params: dict[str, Any],
        *,
        breakdown: dict[str, Any],
        gate: dict[str, Any] | None,
        round_no: int = 1,
    ) -> Proposal:
        rule = self.fallback.propose(params, breakdown=breakdown, gate=gate, round_no=round_no)
        if not rule.give_up:
            return rule            # 规则能处理就不花钱问模型
        if self.llm is None:
            return rule

        import json

        try:
            out = self.llm.structured(
                system=self.SYSTEM,
                user=(
                    f"当前参数（只有 {list(TUNABLE)} 可调）：\n"
                    f"{json.dumps({k: params.get(k) for k in TUNABLE}, ensure_ascii=False)}\n\n"
                    f"质量分：{breakdown.get('score')}\n"
                    f"扣分项：{json.dumps(breakdown.get('deductions', []), ensure_ascii=False)}\n\n"
                    "给出下一轮的参数。如果你认为调参救不了，把 give_up 设为 true。"
                ),
                schema=self.SCHEMA,
                max_retries=2,
            )
        except Exception as e:  # noqa: BLE001 - 模型失败就退回规则的结论
            rule.give_up_reason += f"（模型兜底也失败了：{e}）"
            return rule

        if out.get("give_up"):
            return Proposal(params=dict(params), give_up=True,
                            give_up_reason=out.get("rationale", "模型认为调参救不了"))

        clean, changes = sanitize(params, out.get("params", {}))
        if not changes:
            rule.give_up_reason += "（模型的提议全部越界或无变化，已丢弃）"
            return rule
        return Proposal(params=clean, changes=changes,
                        rationale=out.get("rationale", "")[:400])


def sanitize(
    current: dict[str, Any], proposed: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """只接受白名单参数的合法改动。

    模型返回什么都行，能落地的只有这里放行的部分。
    """
    out = dict(current)
    changes: list[dict[str, Any]] = []
    for k, v in (proposed or {}).items():
        if k not in TUNABLE:
            continue
        try:
            if k == "num_records":
                v = int(v)
                if not (1 <= v <= 2_000_000):
                    continue
            else:
                v = float(v)
                if not (0.0 <= v <= 1.0):
                    continue
        except (TypeError, ValueError):
            continue
        if v != current.get(k):
            changes.append({"param": k, "from": current.get(k), "to": v, "why": "模型提议"})
            out[k] = v
    return out, changes


def assert_no_threshold_drift(before: dict[str, Any], after: dict[str, Any]) -> None:
    """确认 Refine 没有动过门禁阈值。

    这是设计文档 §4.2 的红线，用断言钉死——比写在文档里靠人自觉可靠。
    """
    b = before.get("thresholds") or {}
    a = after.get("thresholds") or {}
    if b != a:
        raise AssertionError(
            f"Refine 试图修改门禁阈值：{b} → {a}。"
            "允许它调阈值的话，循环一定会「收敛」，因为它会把标准降到自己够得着为止。"
        )
