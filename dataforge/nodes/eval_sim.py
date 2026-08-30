"""模拟 Agent，用来测**测评集本身有没有区分度**。

这解决的是一个先有鸡还是先有蛋的问题：
要知道一套测评题好不好，得先有强弱不同的 Agent 去做；
但真跑模型既慢又贵，而且模型今天心情不同结果就不同。

所以这里造几个**行为已知**的假 Agent：

============ ==================================================
perfect      正确完成，且遵守约束
sloppy       任务做对了，但违反约束（多写一次、删了不该删的）
lazy         只读不写 / 只调一个工具就交差
wrong        流程对但算错了
noop         什么都不做
============ ==================================================

**如果这五个的通过率一样，说明测评集没有区分度。**
这不是在测 Agent，是在测题。

⚠ 局限必须说清楚：模拟 Agent 的行为是我按题面写的，
所以它们"知道"正确答案。这测的是**判据能不能区分不同行为**，
不是"真实模型觉得这题难不难"。后者只能用真模型跑（见 inspect_export）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .agenteval import run_assertions

PROFILES = ("perfect", "sloppy", "lazy", "wrong", "noop")


def _expected(task: dict[str, Any], kind: str, key: str = "value") -> Any:
    for a in task.get("assertions", []):
        if a["kind"] == kind:
            return a.get(key)
    return None


def _paths(task: dict[str, Any]) -> list[str]:
    return [a["path"] for a in task.get("assertions", [])
            if a["kind"] in ("file_exists", "file_contains", "file_json_path")]


def simulate(task: dict[str, Any], profile: str, workdir: Path) -> dict[str, Any]:
    """按能力档位模拟一次运行，落地副作用（写文件）并返回结果。

    每个档位都**只做该档位会做的事**——不偷看断言去凑答案，
    否则区分度就是假的。
    """
    if profile not in PROFILES:
        raise ValueError(f"未知档位 {profile!r}，可选：{PROFILES}")

    workdir.mkdir(parents=True, exist_ok=True)
    # 铺初始文件
    for name, content in (task.get("setup", {}).get("files") or {}).items():
        (workdir / name).parent.mkdir(parents=True, exist_ok=True)
        (workdir / name).write_text(str(content), encoding="utf-8")

    calls: list[dict[str, str]] = []
    result: dict[str, Any] = {"output": "", "tool_calls": calls, "error": None}

    def call(name: str) -> None:
        calls.append({"name": name})

    if profile == "noop":
        return result

    tools = {t["name"] for t in (task.get("tools") or [])}

    # 读文件：除了 lazy 之外都会读（lazy 也读，它的问题在于不写）
    if "read_file" in tools:
        call("read_file")

    if profile == "lazy":
        # 读了就交差，不产出任何东西
        result["output"] = "我看了一下，内容大致如此。"
        return result

    # ── 产出阶段 ──
    file_targets = _paths(task)
    json_expect = _expected(task, "file_json_path")
    json_key = _expected(task, "file_json_path", "json_path")
    contains = [(a["path"], a["value"]) for a in task.get("assertions", [])
                if a["kind"] == "file_contains"]

    # wrong 档必须**处处都错**，不能只错在数值上。
    # 只扰动 value_equals 的话，测出来的是模拟器的偷懒，
    # 不是测评集抓不抓得住错误答案。
    def spoil(v: Any) -> Any:
        if isinstance(v, bool):
            return not v
        if isinstance(v, (int, float)):
            return v + 1
        return f"{v}_错的"

    is_wrong = profile == "wrong"

    if "write_file" in tools and file_targets:
        # 需要写的目标：优先写 file_contains / file_json_path 指定的那个
        for path, value in contains:
            (workdir / path).parent.mkdir(parents=True, exist_ok=True)
            existing = (workdir / path).read_text(encoding="utf-8") \
                if (workdir / path).exists() else ""
            payload = "内容写错了" if is_wrong else value
            if payload not in existing:
                (workdir / path).write_text(f"{existing}\n{payload}".strip(), encoding="utf-8")
            call("write_file")
        if json_key is not None:
            target = next(a["path"] for a in task["assertions"]
                          if a["kind"] == "file_json_path")
            (workdir / target).write_text(
                json.dumps({json_key: spoil(json_expect) if is_wrong else json_expect},
                           ensure_ascii=False), encoding="utf-8")
            call("write_file")
        # 只要求存在、没要求内容的文件
        for p in file_targets:
            if not (workdir / p).exists():
                (workdir / p).write_text("已处理", encoding="utf-8")
                call("write_file")

    # ── 计算类 ──
    num_expect = _expected(task, "value_equals")
    if num_expect is not None and "calculator" in tools:
        call("calculator")
        result["result"] = spoil(num_expect) if is_wrong else num_expect

    # sloppy：任务做完了，但不该做的也做了
    if profile == "sloppy":
        if "delete_file" in tools:
            call("delete_file")
            for p in (task.get("setup", {}).get("files") or {}):
                fp = workdir / p
                if fp.exists() and p not in file_targets:
                    fp.unlink()
        elif "write_file" in tools and file_targets:
            # 没有 delete_file 可违反时，改为多写一次（违反"只写一次"的约束）
            (workdir / file_targets[0]).write_text("又写了一遍", encoding="utf-8")
            call("write_file")

    # constrained 类会检查调用计数
    result["write_file_calls"] = sum(1 for c in calls if c["name"] == "write_file")
    result["output"] = "任务已完成。"
    return result


def sweep(tasks: list[dict[str, Any]], root: Path,
          profiles: tuple[str, ...] = PROFILES) -> dict[str, Any]:
    """让每个档位跑完全部任务，统计通过率。

    Returns:
        含总体通过率、按难度、按类别、以及**区分度**的报告。
    """
    rows: dict[str, dict[str, Any]] = {}
    detail: list[dict[str, Any]] = []

    for prof in profiles:
        passed = 0
        by_diff: dict[str, list[int]] = {}
        by_cat: dict[str, list[int]] = {}
        for i, t in enumerate(tasks):
            wd = root / prof / f"t{i:04d}"
            if wd.exists():
                import shutil
                shutil.rmtree(wd)
            res = simulate(t, prof, wd)
            ok = run_assertions(t, res, workdir=wd)["passed"]
            passed += int(ok)
            by_diff.setdefault(t.get("difficulty", "?"), []).append(int(ok))
            by_cat.setdefault(t.get("category", "?"), []).append(int(ok))
            detail.append({"profile": prof, "task_id": t["task_id"],
                           "difficulty": t.get("difficulty"), "passed": ok})
        rows[prof] = {
            "pass_rate": round(passed / max(1, len(tasks)), 3),
            "passed": passed, "total": len(tasks),
            "by_difficulty": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_diff.items())},
            "by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
        }

    rates = [rows[p]["pass_rate"] for p in profiles]
    return {
        "profiles": rows,
        "spread": round(max(rates) - min(rates), 3),
        "discriminates": max(rates) - min(rates) >= 0.3,
        "detail": detail,
        "note": (
            "模拟 Agent 的行为是按题面写死的，它们「知道」正确答案。"
            "这测的是判据能不能区分不同行为，**不是真实模型觉得这题难不难**。"
        ),
    }
