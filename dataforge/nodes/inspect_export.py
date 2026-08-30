"""把 DataForge 测评集导出成 Inspect（UK AISI）能跑的格式。

为什么要导出而不是自建 runner：
DataForge 的自测只能回答「判据能不能区分不同行为」，
回答不了「**真实模型觉得这题难不难**」。后者必须用真模型跑，
而 Inspect 是英国 AISI 做前沿模型上线前安全测试用的框架，
接它比自己写一个 runner 靠谱。

产出两个文件：

``tasks.jsonl``   每行一个 Inspect ``Sample``（含 files / metadata）
``eval_task.py``  Inspect ``@task``，内含一个调用 DataForge 断言的 scorer

**判对错的仍然是 DataForge 的断言**，Inspect 只负责跑模型、给沙箱、收日志。
不把判据交给 Inspect 的 model-graded scorer —— 那就退回 LLM-as-Judge 了。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

V = {"export_inspect": "1.0.0"}


def task_to_sample(task: dict[str, Any]) -> dict[str, Any]:
    """DataForge 任务 → Inspect Sample 的 JSON 形态。

    ``target`` 在 Inspect 里是"理想输出"，但我们的判据是断言而不是文本比对，
    所以放一个标记值，真正的判据走 ``metadata.assertions``。
    """
    return {
        "id": task["task_id"],
        "input": task["prompt"],
        # 断言式评分不用 target 比对；放标记而不是留空，
        # 免得有人以为忘填了
        "target": "ASSERTION_SCORED",
        "files": {k: str(v) for k, v in
                  (task.get("setup", {}).get("files") or {}).items()},
        "metadata": {
            "assertions": task["assertions"],
            "tools": [t["name"] for t in (task.get("tools") or [])],
            "category": task.get("category"),
            "difficulty": task.get("difficulty"),
        },
    }


SCORER_TEMPLATE = '''"""由 DataForge 自动生成的 Inspect 任务。

    inspect eval eval_task.py --model openai/gpt-4o

判对错的是 DataForge 的断言（``dataforge.nodes.agenteval.run_assertions``），
不是 model-graded scorer —— 判据必须是代码。

生成时间：{generated_at}
任务数：{n_tasks}
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, basic_agent
from inspect_ai.tool import bash, python
from inspect_ai.util import sandbox

from dataforge.nodes.agenteval import run_assertions

HERE = Path(__file__).parent


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["id"],
        input=record["input"],
        target=record["target"],
        files=record.get("files") or {{}},
        metadata=record.get("metadata") or {{}},
    )


async def _collect_result(state: TaskState) -> dict:
    """把 Inspect 的执行状态整理成 DataForge 断言认识的形状。"""
    calls = []
    for msg in state.messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            calls.append({{"name": getattr(tc, "function", None) or str(tc)}})
    result = {{
        "output": state.output.completion if state.output else "",
        "tool_calls": calls,
        "error": None,
        "write_file_calls": sum(1 for c in calls if "write" in (c["name"] or "")),
    }}
    # 模型若以 JSON 形式给出结构化结果，取出来供 value_* 断言使用
    try:
        payload = json.loads((state.output.completion or "").strip())
        if isinstance(payload, dict):
            result.update({{k: v for k, v in payload.items()
                            if k not in ("tool_calls", "output")}})
    except Exception:
        pass
    return result


@scorer(metrics=[accuracy(), stderr()])
def dataforge_assertions():
    """用 DataForge 的断言判定成败。全部断言通过才算 1 分。"""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {{}}
        task_def = {{"task_id": str(state.sample_id),
                     "assertions": meta.get("assertions", [])}}

        # 文件类断言要读沙箱里的文件，先落到本地临时目录
        workdir = HERE / "_sandbox" / str(state.sample_id)
        workdir.mkdir(parents=True, exist_ok=True)
        for a in task_def["assertions"]:
            p = a.get("path")
            if not p:
                continue
            try:
                content = await sandbox().read_file(p)
            except Exception:
                continue
            (workdir / p).parent.mkdir(parents=True, exist_ok=True)
            (workdir / p).write_text(
                content if isinstance(content, str) else content.decode("utf-8", "replace"),
                encoding="utf-8")

        result = await _collect_result(state)
        out = run_assertions(task_def, result, workdir=workdir)
        return Score(
            value=1 if out["passed"] else 0,
            answer=result["output"][:200],
            explanation="; ".join(
                c["message"] for c in out["checks"] if not c["passed"]) or "全部断言通过",
            metadata={{"checks": out["checks"],
                       "difficulty": meta.get("difficulty"),
                       "category": meta.get("category")}},
        )

    return score


@task
def dataforge_eval():
    return Task(
        dataset=json_dataset(str(HERE / "tasks.jsonl"), _record_to_sample),
        solver=basic_agent(tools=[bash(), python()], max_attempts=1),
        scorer=dataforge_assertions(),
        sandbox="docker",
    )
'''


def export(tasks: list[dict[str, Any]], out_dir: str | Path,
           *, generated_at: str = "") -> dict[str, Any]:
    """把测评集写成 Inspect 可跑的目录。

    Returns:
        含文件路径与统计的字典。
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    samples = [task_to_sample(t) for t in tasks]
    jsonl = d / "tasks.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8")

    py = d / "eval_task.py"
    py.write_text(
        SCORER_TEMPLATE.format(generated_at=generated_at or "-", n_tasks=len(tasks)),
        encoding="utf-8")

    readme = d / "README.md"
    readme.write_text(_readme(len(tasks), tasks), encoding="utf-8")

    return {
        "dir": str(d),
        "tasks_file": str(jsonl),
        "task_py": str(py),
        "readme": str(readme),
        "num_samples": len(samples),
        "with_files": sum(1 for s in samples if s["files"]),
        "by_difficulty": _count(tasks, "difficulty"),
        "by_category": _count(tasks, "category"),
        "hint": (
            "在该目录下运行：inspect eval eval_task.py --model <你的模型>。"
            "需要 Docker（Inspect 的沙箱依赖它），以及 pip install inspect-ai。"
        ),
    }


def _count(tasks: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        out[t.get(key, "?")] = out.get(t.get(key, "?"), 0) + 1
    return out


def _readme(n: int, tasks: list[dict[str, Any]]) -> str:
    diff = _count(tasks, "difficulty")
    cat = _count(tasks, "category")
    return f"""# DataForge 测评集（Inspect 格式）

{n} 个任务 · 难度 {diff} · 类别 {cat}

## 跑起来

```bash
pip install inspect-ai
inspect eval eval_task.py --model openai/gpt-4o
```

需要 **Docker**——Inspect 的沙箱依赖它。

## 判对错的是谁

**是 DataForge 的断言，不是模型。** `eval_task.py` 里的 scorer 调用
`dataforge.nodes.agenteval.run_assertions`，逐条执行
文件内容匹配、工具是否被调用、数值是否相等等确定性检查。

刻意**没有**用 Inspect 的 model-graded scorer——那会退回 LLM-as-Judge，
而评分漂移、对提示词敏感、无法回归正是要避免的。

## 这些数据是合成的

任务与初始文件都是按模板生成的合成数据，**不得用于刷榜**，
也不得作为真实基准评测结果报告。

## 已知局限

- 任务模板只有 5 类，多样性有限
- 难度分层（easy/medium/hard）是**按"需要多做对几件事"划的**，
  尚未用真实模型验证过它是否真的构成梯度——这正是跑这套测评要回答的问题
"""
