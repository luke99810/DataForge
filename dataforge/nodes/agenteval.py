"""Agent 测评集生成。

设计文档 §7 的核心要求：

    **每个测评样本必须带一个机器可判的成功条件。**

    纯 LLM-as-Judge 的测评集是不可靠的：评分会漂移、对提示词敏感、无法回归。
    LLM-as-Judge 只用于主观维度的**排序**（哪个回答更清楚），
    **不用于判定对错**。判对错的必须是代码。

所以这里的重点不是"生成很多任务"，而是**断言能不能真的跑起来**。
一条跑不起来的断言比没有断言更糟——它给人一种"有判据"的错觉。

断言是一个受限的小 DSL，故意不支持任意代码：
测评集会被别人拿去跑，里面能塞任意代码就是一个远程执行漏洞。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..graph.node import NodeContext

V = {
    "generate_agent_eval": "1.0.0",
    "agent_eval_gate": "1.0.0",
}

# 支持的断言类型。**刻意不含 "python" / "eval" / "shell"**：
# 测评集是要发出去给别人跑的，能塞任意代码就是远程执行漏洞。
ASSERT_KINDS = (
    "file_exists",       # 某个文件存在
    "file_contains",     # 文件内容包含某串（可正则）
    "file_json_path",    # 文件的 JSON 某路径等于期望值
    "value_equals",      # 结果中某字段等于期望值
    "value_close",       # 数值在容差内
    "value_in",          # 取值属于集合
    "tool_called",       # 某工具被调用过（可带次数下限）
    "tool_not_called",   # 某工具不该被调用（安全类任务的关键判据）
    "output_matches",    # 最终输出匹配正则
    "no_exception",      # 全程未抛异常
)


class AssertionError_(ValueError):
    """断言定义本身有问题（不是断言失败）。"""


# ── 断言执行 ──────────────────────────────────────────────

def _get_path(obj: Any, path: str) -> Any:
    """按 ``a.b.0.c`` 取值。取不到返回哨兵。"""
    cur = obj
    for part in str(path).split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return _MISSING
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        else:
            return _MISSING
    return cur


class _Missing:
    def __repr__(self) -> str:
        return "<缺失>"


_MISSING = _Missing()


def check_assertion(a: dict[str, Any], result: dict[str, Any], *,
                    workdir: str | Path | None = None) -> dict[str, Any]:
    """执行一条断言，返回 ``{passed, message}``。

    Args:
        a: 断言定义。
        result: 被测 Agent 的运行结果，形如
            ``{"output": str, "tool_calls": [{"name":...}], "error": str|None}``。
        workdir: 文件类断言的根目录。
    """
    kind = a.get("kind")
    if kind not in ASSERT_KINDS:
        raise AssertionError_(f"未知断言类型 {kind!r}，可选：{ASSERT_KINDS}")

    root = Path(workdir) if workdir else Path(".")

    def ok(msg: str) -> dict[str, Any]:
        return {"kind": kind, "passed": True, "message": msg}

    def no(msg: str) -> dict[str, Any]:
        return {"kind": kind, "passed": False, "message": msg}

    if kind in ("file_exists", "file_contains", "file_json_path"):
        target = root / a["path"]
        if not target.exists():
            return no(f"文件不存在：{a['path']}")
        if kind == "file_exists":
            return ok(f"文件存在：{a['path']}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if kind == "file_contains":
            needle = a["value"]
            hit = (re.search(needle, text) is not None) if a.get("regex") else (needle in text)
            return ok(f"{a['path']} 命中 {needle!r}") if hit else \
                no(f"{a['path']} 里找不到 {needle!r}")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            return no(f"{a['path']} 不是合法 JSON：{e}")
        got = _get_path(doc, a["json_path"])
        return ok(f"{a['json_path']} == {got!r}") if got == a["value"] else \
            no(f"{a['json_path']} 期望 {a['value']!r}，实际 {got!r}")

    if kind == "value_equals":
        got = _get_path(result, a["path"])
        return ok(f"{a['path']} == {got!r}") if got == a["value"] else \
            no(f"{a['path']} 期望 {a['value']!r}，实际 {got!r}")

    if kind == "value_close":
        got = _get_path(result, a["path"])
        tol = float(a.get("tolerance", 1e-6))
        try:
            diff = abs(float(got) - float(a["value"]))
        except (TypeError, ValueError):
            return no(f"{a['path']} 不是数值：{got!r}")
        return ok(f"{a['path']} 与期望相差 {diff:.3g} ≤ {tol}") if diff <= tol else \
            no(f"{a['path']} 期望 {a['value']}±{tol}，实际 {got}（差 {diff:.3g}）")

    if kind == "value_in":
        got = _get_path(result, a["path"])
        return ok(f"{a['path']} = {got!r} 在允许集合内") if got in a["value"] else \
            no(f"{a['path']} = {got!r} 不在 {a['value']!r} 内")

    if kind in ("tool_called", "tool_not_called"):
        calls = [c.get("name") for c in (result.get("tool_calls") or [])]
        n = calls.count(a["name"])
        if kind == "tool_called":
            need = int(a.get("min_times", 1))
            return ok(f"{a['name']} 被调用 {n} 次") if n >= need else \
                no(f"{a['name']} 至少应调用 {need} 次，实际 {n} 次")
        return ok(f"{a['name']} 未被调用") if n == 0 else \
            no(f"{a['name']} 不该被调用，却调用了 {n} 次")

    if kind == "output_matches":
        text = str(result.get("output", ""))
        return ok("输出匹配") if re.search(a["value"], text) else \
            no(f"输出不匹配 {a['value']!r}")

    # no_exception
    err = result.get("error")
    return ok("全程无异常") if not err else no(f"抛了异常：{err}")


def run_assertions(task: dict[str, Any], result: dict[str, Any], *,
                   workdir: str | Path | None = None) -> dict[str, Any]:
    """跑完一个任务的全部断言。

    **全部通过才算成功。** 不做「通过一半算半分」——
    测评的意义在于给出确定的对错，模糊化就退回 LLM-as-Judge 了。
    """
    checks = [check_assertion(a, result, workdir=workdir) for a in task.get("assertions", [])]
    return {
        "task_id": task.get("task_id"),
        "passed": bool(checks) and all(c["passed"] for c in checks),
        "checks": checks,
        "failed": [c for c in checks if not c["passed"]],
    }


# ── 断言定义校验 ──────────────────────────────────────────

_REQUIRED_KEYS = {
    "file_exists": ("path",),
    "file_contains": ("path", "value"),
    "file_json_path": ("path", "json_path", "value"),
    "value_equals": ("path", "value"),
    "value_close": ("path", "value"),
    "value_in": ("path", "value"),
    "tool_called": ("name",),
    "tool_not_called": ("name",),
    "output_matches": ("value",),
    "no_exception": (),
}


def check_assertion_def(a: dict[str, Any]) -> list[str]:
    """校验断言**定义**本身。

    一条跑不起来的断言比没有断言更糟——它给人「有判据」的错觉，
    实际跑测评时会静默跳过或炸掉。所以定义阶段就要拦。
    """
    issues: list[str] = []
    kind = a.get("kind")
    if kind not in ASSERT_KINDS:
        return [f"未知断言类型 {kind!r}"]
    for k in _REQUIRED_KEYS[kind]:
        if k not in a:
            issues.append(f"{kind} 缺少必填项 {k!r}")
    if kind in ("output_matches",) or (kind == "file_contains" and a.get("regex")):
        try:
            re.compile(a.get("value", ""))
        except re.error as e:
            issues.append(f"{kind} 的正则不合法：{e}")
    if kind == "value_in" and not isinstance(a.get("value"), (list, tuple)):
        issues.append("value_in 的 value 必须是列表")
    if kind == "value_close":
        try:
            float(a.get("value"))
        except (TypeError, ValueError):
            issues.append("value_close 的 value 必须是数值")
    return issues


def check_task(task: dict[str, Any]) -> list[str]:
    """校验一个测评任务。"""
    issues: list[str] = []
    if not task.get("task_id"):
        issues.append("缺少 task_id")
    if not str(task.get("prompt", "")).strip():
        issues.append("缺少任务描述 prompt")

    asserts = task.get("assertions") or []
    if not asserts:
        # 这是本模块最重要的一条规则
        issues.append("没有任何断言 —— 每个样本必须有机器可判的成功条件，"
                      "不能只靠 LLM-as-Judge")
    for i, a in enumerate(asserts):
        issues += [f"assertions[{i}]: {m}" for m in check_assertion_def(a)]

    tools = {t.get("name") for t in (task.get("tools") or [])}
    for i, a in enumerate(asserts):
        if a.get("kind") in ("tool_called", "tool_not_called") and a.get("name") not in tools:
            issues.append(
                f"assertions[{i}]: 断言引用了工具 {a.get('name')!r}，"
                f"但它不在本任务的工具集里：{sorted(tools)}"
            )
    return issues


# ── 生成 ──────────────────────────────────────────────────

from .eval_templates import DIFFICULTIES, TEMPLATES as _TEMPLATES, derive


def _fill(obj: Any, mapping: dict[str, Any]) -> Any:
    """把 ``{name}`` 占位替换成具体值，保留非字符串类型。"""
    if isinstance(obj, str):
        # 整串就是一个占位时，保留原类型（数字不该变成字符串）
        m = re.fullmatch(r"\{(\w+)\}", obj)
        if m and m.group(1) in mapping:
            return mapping[m.group(1)]
        out = obj
        for k, v in mapping.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(obj, dict):
        return {_fill(k, mapping): _fill(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(x, mapping) for x in obj]
    return obj


def generate_agent_eval(inputs: dict[str, Any], params: dict[str, Any],
                        ctx: NodeContext) -> dict:
    """生成 Agent 测评集。

    每个样本 = 任务描述 + 工具集 + 初始状态 + **机器可判的断言**。
    """
    import numpy as np

    n = int(params.get("num_tasks", 20))
    cats = params.get("categories") or [t["category"] for t in _TEMPLATES]
    diffs = params.get("difficulties") or list(DIFFICULTIES)
    bad = [d for d in diffs if d not in DIFFICULTIES]
    if bad:
        raise ValueError(f"未知难度 {bad}，可选：{DIFFICULTIES}")
    pool = [t for t in _TEMPLATES
            if t["category"] in cats and t["difficulty"] in diffs]
    if not pool:
        raise ValueError(
            f"没有匹配的任务：类别 {cats} × 难度 {diffs}。"
            f"可选类别 {sorted({t['category'] for t in _TEMPLATES})}，"
            f"可选难度 {list(DIFFICULTIES)}")

    rng = np.random.default_rng(int(params.get("seed", ctx.seed if ctx.seed is not None else 42)))
    tasks: list[dict[str, Any]] = []
    for i in range(n):
        # 轮转取模板而不是随机抽：随机抽在小样本下会漏掉整档难度，
        # 那样"有没有区分度"根本测不出来
        tpl = pool[i % len(pool)]
        mapping = {k: v[int(rng.integers(len(v)))] for k, v in tpl["vars"].items()}
        mapping = derive(tpl, mapping)
        task = {
            "task_id": f"{tpl['category']}_{tpl['difficulty']}_{i:04d}",
            "category": tpl["category"],
            "difficulty": tpl["difficulty"],
            "prompt": _fill(tpl["prompt"], mapping),
            "tools": _fill(tpl["tools"], mapping),
            "setup": _fill(tpl.get("setup", {}), mapping),
            "assertions": _fill(tpl["assertions"], mapping),
        }
        issues = check_task(task)
        if issues:
            raise ValueError(f"生成的任务不合法（模板 {tpl['category']} 有问题）：{issues}")
        tasks.append(task)

    by_cat: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    for t in tasks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1
    n_asserts = sum(len(t["assertions"]) for t in tasks)
    ctx.note(f"生成 {len(tasks)} 个测评任务、{n_asserts} 条断言", by_category=by_cat)

    return {
        "tasks": tasks,
        "stats": {
            "num_tasks": len(tasks),
            "num_assertions": n_asserts,
            "assertions_per_task": round(n_asserts / max(1, len(tasks)), 2),
            "by_category": by_cat,
            "by_difficulty": by_diff,
            "assertion_kinds": sorted({a["kind"] for t in tasks for a in t["assertions"]}),
        },
        "notice": (
            "每个任务的成功判据都是机器可判的。LLM-as-Judge 只应用于主观维度排序，"
            "不应用于判定对错。"
        ),
    }


def agent_eval_gate(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """测评集门禁 —— 断言跑不起来就阻断。

    这里做的不只是「定义看起来对」，还**用桩结果真的跑一遍断言**：
    一条永远为真、或永远报错的断言，都区分不出好坏 Agent。
    """
    from ..graph.executor import NodeBlocked

    data = inputs["eval_set"]
    tasks = data.get("tasks") or []
    if not tasks:
        raise NodeBlocked("[agent_eval] 测评集是空的", gate="agent_eval",
                          details={"all": []})

    def_issues: list[str] = []
    for t in tasks:
        def_issues += [f"{t.get('task_id')}: {m}" for m in check_task(t)]

    # 用一个"什么都没做"的桩结果跑一遍：绝大多数断言应当失败。
    # 全都通过 = 判据没有区分力。
    empty = {"output": "", "tool_calls": [], "error": None}
    trivially_passing: list[str] = []
    crashed: list[str] = []
    for t in tasks:
        try:
            r = run_assertions(t, empty, workdir=params.get("workdir"))
            if r["passed"]:
                trivially_passing.append(t["task_id"])
        except Exception as e:  # noqa: BLE001
            crashed.append(f"{t['task_id']}: {type(e).__name__}: {e}")

    findings = {
        "definition_issues": def_issues,
        "trivially_passing": trivially_passing,
        "crashed": crashed,
    }
    errors: list[str] = []
    if def_issues:
        errors.append(f"{len(def_issues)} 条断言定义有问题")
    if crashed:
        errors.append(f"{len(crashed)} 个任务的断言执行时报错")
    if trivially_passing:
        errors.append(
            f"{len(trivially_passing)} 个任务在「什么都不做」时也能通过 —— "
            "这种判据区分不出好坏 Agent"
        )

    ctx.note(
        f"测评集检查：{len(tasks)} 个任务，"
        f"定义问题 {len(def_issues)}，空跑通过 {len(trivially_passing)}，报错 {len(crashed)}"
    )
    if errors:
        raise NodeBlocked(
            "[agent_eval] " + "；".join(errors),
            gate="agent_eval",
            details={"all": [{"gate": "agent_eval", "level": "BLOCK", "passed": False,
                              "message": "；".join(errors), "details": findings}]},
        )
    return {
        "passed": True,
        "results": [{"gate": "agent_eval", "level": "BLOCK", "passed": True,
                     "message": f"{len(tasks)} 个任务的断言均可执行且有区分力",
                     "details": findings}],
        "audit": {"thresholds": {}, "overrides": {}, "relaxed": False},
        "stats": data.get("stats", {}),
    }
