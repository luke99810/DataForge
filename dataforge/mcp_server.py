"""DataForge 作为 MCP 服务端。

设计文档 §9：让 Claude Code / Cursor 等直接调用 DataForge，
使它成为生态里的**能力提供方**，而不只是一个自己用的工具。

    dataforge-mcp                       # stdio，供 MCP 客户端拉起
    python -m dataforge.mcp_server      # 同上

三条红线在这一层同样成立，而且要更小心——调用方是另一个 Agent，
不会像人那样看一眼再点确认：

1. **合成标记不可关闭。** 每个产出的数据集都带 ``synthetic: true``
   与完整血缘，返回文本里也明写「不得用于刷榜或充当真实评测结果」。
2. **门禁阈值不接受调用方指定。** 允许外部传阈值，等于把
   「把标准降到自己够得着为止」这条路重新打开。
3. **人工确认闸门保留。** ``parse_paper`` 只返回变量卡；
   要拿到数据集，调用方**必须显式列出采纳哪些变量**——
   拿不到「一句话直接出数据集」的捷径。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core.artifacts import ArtifactRef, ArtifactStore
from .core.runstore import RunStore
from .graph.executor import Executor
from .harness.score import gate_doc_of, score_run
from .llm.client import from_env
from .pipelines import (
    agent_eval_pipeline, llm_sft_pipeline, paper_pipeline, rl_pipeline,
    source_aligned_pipeline, tabular_pipeline, timeseries_pipeline,
)
from .sources import HttpFetcher, SourceError, known as known_benchmarks, resolve as resolve_link
from .planner import PlanError, Planner, plan_to_request

# 刻意不用 ⚠ 之类的符号：这段文字会流向各种下游工具，
# 而 Windows 控制台默认 GBK，非常用符号会直接抛 UnicodeEncodeError。
# 符号对语义没有增益，去掉它换来的是少一类编码麻烦。
SYNTHETIC_NOTICE = "本数据集为合成数据。不得用于刷榜，也不得作为真实评测结果报告。"


# ── 工作区 ────────────────────────────────────────────────

class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.root / "artifacts")
        self.runstore = RunStore(self.root / "runs.sqlite")
        self.papers = self.root / "papers"
        self.papers.mkdir(parents=True, exist_ok=True)
        self.llm = from_env()
        # 取数器只在被显式允许时才真正出网（见 df_align_to_source 的 allow_network）
        self.fetcher = HttpFetcher()

    def executor(self) -> Executor:
        return Executor(self.store, self.runstore, workdir=self.root,
                        llm=self.llm, fetcher=self.fetcher)

    def close(self) -> None:
        self.runstore.close()


# ── 工具实现（与 MCP 协议解耦，便于单测） ────────────────

def _summarize(rep: Any, ws: Workspace) -> dict[str, Any]:
    """把一次运行压成调用方用得上的摘要。

    不回传数据本体——几十万行塞进对话上下文毫无意义。
    回传的是产物引用，需要时再用 ``fetch_artifact`` 取。
    """
    bd = score_run(rep.results, ws.store)
    gate = gate_doc_of(rep.results.get("gate"), ws.store)
    out: dict[str, Any] = {
        "run_id": rep.run_id,
        "status": rep.status,
        "quality_score": round(bd.score, 2),
        "nodes": {
            nid: {
                "status": r.status.value,
                "artifact": r.artifact.to_dict() if r.artifact else None,
                "error": r.error,
            }
            for nid, r in rep.results.items()
        },
    }
    if bd.deductions:
        out["deductions"] = bd.deductions
    if gate:
        out["gate_passed"] = gate.get("passed")
        out["gate_results"] = [
            {"gate": g["gate"], "level": g["level"], "passed": g["passed"],
             "message": g["message"]}
            for g in gate.get("results", [])
        ]
    if rep.status == "success":
        out["notice"] = SYNTHETIC_NOTICE
    return out


def tool_generate(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """按字段定义生成数据集。"""
    fields = args.get("fields") or []
    if not fields:
        raise ValueError("至少需要一个字段")
    kind = args.get("kind", "tabular")

    common = dict(
        fields=fields,
        num_records=int(args.get("num_records", 1000)),
        name=args.get("name", "dataset"),
        target=args.get("target"),
        # 阈值刻意不开放给调用方：见模块开头红线 2
        thresholds={},
        known_correlated=[list(p) for p in (args.get("known_correlated") or [])],
    )
    if kind == "timeseries":
        tcol = args.get("time_column")
        if not tcol:
            raise ValueError("时序流水线必须指定 time_column")
        dag = timeseries_pipeline(time_column=tcol, **common)
    else:
        dag = tabular_pipeline(
            task_type=args.get("task_type", "classification"),
            missing_rate=float(args.get("missing_rate", 0.0)),
            noise_ratio=float(args.get("noise_ratio", 0.0)),
            **common,
        )

    rep = ws.executor().run(dag, seed=int(args.get("seed", 20260823)), label="mcp")
    return _summarize(rep, ws)


def tool_plan(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """自然语言 → 计划书。**只规划，不执行。**"""
    planner = Planner(ws.llm, runstore=ws.runstore, store=ws.store)
    plan, request = planner.plan_request(
        args["text"], seed=int(args.get("seed", 20260823))
    )
    return {
        "plan": plan.spec,
        "issues": plan.issues,
        "fixes": plan.fixes,
        "cached": plan.cached,
        "runnable_args": request,
        "hint": (
            "计划书已就绪，检查无误后把 runnable_args 传给 df_generate 即可执行。"
            if request else
            "计划书有语义问题（见 issues），修正后再执行。"
        ),
    }


def tool_parse_paper(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """解析论文并抽取变量卡。**只到变量卡为止。**

    不直接产出数据集：人工确认闸门对 MCP 调用方同样成立，
    要生成数据必须再调 ``df_dataset_from_paper`` 并显式列出采纳的变量。
    """
    path = Path(args["path"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"论文不存在：{path}")

    dag = paper_pipeline(pdf_path=str(path), name=args.get("name", "paper_dataset"))
    rep = ws.executor().run(dag, seed=1, targets=["extract"], label="mcp-paper")
    r = rep.results.get("extract")
    if r is None or r.artifact is None:
        raise RuntimeError(
            f"抽取失败：{(r.error if r else '未执行')}"
        )
    ex = ws.store.get(r.artifact)
    return {
        "paper": str(path),
        "task_type": ex.get("task_type"),
        "variables": ex.get("variables", []),
        "open_questions": ex.get("open_questions", []),
        "suggested_adopt": [v["name"] for v in ex.get("variables", []) if v.get("suggest_adopt")],
        "gate": (
            "这是人工确认闸门：变量抽错了后面全错且看不出来。"
            "请核对每张卡的 source_span（论文原文位置），"
            "再调 df_dataset_from_paper 并在 adopt 里显式列出要采纳的变量。"
            "标了 from_related_work 的来自「相关工作」章节，通常不该采纳。"
        ),
    }


def tool_dataset_from_paper(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """在确认变量之后，由论文产出数据集。"""
    adopt = args.get("adopt")
    if not adopt:
        raise ValueError(
            "必须显式列出采纳的变量（adopt）。"
            "先调 df_parse_paper 看过变量卡与原文出处再决定——这一步不能跳。"
        )
    path = Path(args["path"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"论文不存在：{path}")

    decisions: dict[str, Any] = {"adopt": list(adopt)}
    if args.get("overrides"):
        decisions["overrides"] = args["overrides"]
    if args.get("answers"):
        decisions["answers"] = args["answers"]

    dag = paper_pipeline(
        pdf_path=str(path),
        name=args.get("name", "paper_dataset"),
        num_records=int(args.get("num_records", 1000)),
        decisions=decisions,
    )
    rep = ws.executor().run(dag, seed=int(args.get("seed", 20260823)), label="mcp-paper")
    out = _summarize(rep, ws)
    out["adopted"] = list(adopt)
    if "confirm" in rep.lineage.entries:
        out["provenance"] = rep.lineage.entries["confirm"].columns
    return out


def tool_rl_dataset(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """生成 RL 轨迹。门禁会**重放动力学**核对每条转移。"""
    dag = rl_pipeline(
        env=args.get("env", "gridworld"),
        env_params=args.get("env_params"),
        num_episodes=int(args.get("num_episodes", 200)),
        max_steps=int(args.get("max_steps", 50)),
        epsilon=float(args.get("epsilon", 0.2)),
    )
    rep = ws.executor().run(dag, seed=int(args.get("seed", 20260823)), label="mcp-rl")
    out = _summarize(rep, ws)
    gen = rep.results.get("generate")
    if gen is not None and gen.artifact is not None:
        data = ws.store.get(gen.artifact)
        out["env_spec"] = data["env_spec"]
        out["trajectory_stats"] = data["stats"]
        out["hint"] = (
            "转移已逐条重放核对。s' 不是从 (s,a) 走出来的数据，"
            "训出来的模型学的是噪声，而统计指标上完全看不出来。"
        )
    return out


def tool_agent_eval_set(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """生成 Agent 测评集。每个任务都带**机器可判**的成功条件。"""
    dag = agent_eval_pipeline(
        num_tasks=int(args.get("num_tasks", 20)),
        categories=args.get("categories"),
    )
    rep = ws.executor().run(dag, seed=int(args.get("seed", 20260823)), label="mcp-eval")
    out = _summarize(rep, ws)
    gen = rep.results.get("generate")
    if gen is not None and gen.artifact is not None:
        data = ws.store.get(gen.artifact)
        out["eval_stats"] = data["stats"]
        out["sample_task"] = data["tasks"][0] if data["tasks"] else None
        out["hint"] = (
            "断言已用空结果试跑，确认有区分力。"
            "LLM-as-Judge 只应用于主观维度排序，不应用于判定对错。"
        )
    return out


def tool_llm_dataset(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """生成 SFT（可选 DPO）训练集。门禁对**基准污染零容忍**。"""
    dag = llm_sft_pipeline(
        num_samples=int(args.get("num_samples", 200)),
        with_dpo=bool(args.get("with_dpo", True)),
        reference=[tuple(x) for x in (args.get("reference") or [])] or None,
    )
    rep = ws.executor().run(dag, seed=int(args.get("seed", 20260823)), label="mcp-llm")
    out = _summarize(rep, ws)
    sft = rep.results.get("sft")
    if sft is not None and sft.artifact is not None:
        out["sft_stats"] = ws.store.get(sft.artifact)["stats"]
    dpo = rep.results.get("dpo")
    if dpo is not None and dpo.artifact is not None:
        out["dpo_stats"] = ws.store.get(dpo.artifact)["stats"]
    out["hint"] = (
        "已做去重与基准污染检查。注意：只自动删完全重复，"
        "近重复只标出不删——合法变体的相似度可能比真重复还高。"
    )
    return out


def tool_run_assertions(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """拿一个测评任务和一次 Agent 运行结果，判定通过与否。

    这是把测评集**真正用起来**的入口：判对错的是代码，不是模型。
    """
    from .nodes.agenteval import run_assertions

    task = args["task"]
    result = args["result"]
    return run_assertions(task, result, workdir=args.get("workdir"))


def tool_inspect_source(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """看一个链接对应什么结构。**只读元数据，不生成任何数据。**"""
    fetcher = ws.fetcher if args.get("allow_network") else None
    out = resolve_link(args["link"], fetcher)
    return {
        "resolved_by": out["resolved_by"],
        "ref": out["ref"],
        "benchmark": out.get("benchmark"),
        "fields": out["fields"],
        "skipped": out["skipped"],
        "splits": out.get("splits"),
        "protocol": out.get("protocol"),
        "provenance": out["provenance"],
        "notice": out.get("notice"),
        "hint": (
            "这里只读了元数据与结构，没有下载任何数据行。"
            "要按这个结构生成同构合成数据，调 df_align_to_source。"
        ),
    }


def tool_align_to_source(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """按外部链接的结构生成**同构**合成数据集。"""
    dag = source_aligned_pipeline(
        link=args["link"],
        num_records=int(args.get("num_records", 1000)),
        name=args.get("name", "aligned_dataset"),
        target=args.get("target"),
        allow_network=bool(args.get("allow_network", False)),
    )
    rep = ws.executor().run(dag, seed=int(args.get("seed", 20260823)), label="mcp-align")
    out = _summarize(rep, ws)
    exp = rep.results.get("export")
    if exp is not None and exp.artifact is not None:
        card = ws.store.get(exp.artifact)
        out["aligned_to"] = card["aligned_to"]
        out["notice"] = card["notice"]
        if card["aligned_to"].get("skipped_fields"):
            out["warning"] = (
                f"有字段映射不了被跳过：{card['aligned_to']['skipped_fields']}。"
                "生成的数据不含这些列。"
            )
    return out


def tool_list_benchmarks(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """列出本地注册表里已知的基准（命中它们不需要出网）。"""
    from .sources import match as match_bench

    return {
        "benchmarks": [match_bench(k) for k in known_benchmarks()],
        "note": "注册表是手工维护的，必然滞后。认不出来的链接会明说，不猜。",
    }


def tool_eval_discrimination(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """用能力档位不同的模拟 Agent 跑测评集，测**这套题有没有区分度**。

    这测的是题，不是 Agent。五个档位通过率一样 = 这套题白出了。
    """
    import tempfile
    from pathlib import Path

    from .nodes.agenteval import generate_agent_eval
    from .nodes.eval_sim import sweep
    from .graph.node import NodeContext

    ctx = NodeContext(node_id="mcp_sweep", run_id="mcp",
                      seed=int(args.get("seed", 20260824)))
    out = generate_agent_eval(
        {}, {"num_tasks": int(args.get("num_tasks", 40)),
             "categories": args.get("categories"),
             "difficulties": args.get("difficulties")}, ctx)
    rep = sweep(out["tasks"], Path(tempfile.mkdtemp()))
    return {
        "num_tasks": len(out["tasks"]),
        "by_difficulty": out["stats"]["by_difficulty"],
        "profiles": {k: {"pass_rate": v["pass_rate"],
                         "by_difficulty": v["by_difficulty"],
                         "by_category": v["by_category"]}
                     for k, v in rep["profiles"].items()},
        "spread": rep["spread"],
        "discriminates": rep["discriminates"],
        "note": rep["note"],
    }


def tool_export_inspect(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """把测评集导出成 Inspect（UK AISI）能跑的格式。

    判对错的仍是 DataForge 的断言，Inspect 只负责跑模型、给沙箱、收日志。
    """
    from .graph.node import NodeContext
    from .nodes.agenteval import generate_agent_eval
    from .nodes.inspect_export import export

    ctx = NodeContext(node_id="mcp_export", run_id="mcp",
                      seed=int(args.get("seed", 20260824)))
    out = generate_agent_eval(
        {}, {"num_tasks": int(args.get("num_tasks", 40)),
             "categories": args.get("categories"),
             "difficulties": args.get("difficulties")}, ctx)
    dest = Path(args["out_dir"]).expanduser()
    return export(out["tasks"], dest, generated_at=args.get("generated_at", ""))


def tool_validate(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """对已有数据跑质量门禁与泄漏检测。"""
    import pandas as pd

    from .quality import leakage as lk

    path = Path(args["path"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(path, lines=suffix == ".jsonl")
    else:
        raise ValueError(f"不支持的格式 {suffix}（支持 csv / parquet / json / jsonl）")

    findings = lk.check_all(
        df, target=args.get("target"),
        known_correlated=[tuple(p) for p in (args.get("known_correlated") or [])],
    )
    return {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "completeness": round(float(1 - df.isna().to_numpy().mean()), 4) if df.size else 0.0,
        "leakage": [f.to_dict() for f in findings],
        "has_error": lk.has_error(findings),
        "hint": (
            "检出 error 级泄漏。泄漏是合成数据最典型的静默故障——"
            "指标好得离谱、上线全崩。改参数救不了，要改切分或特征。"
            if lk.has_error(findings) else "未检出 error 级泄漏。"
        ),
    }


def tool_analyze(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """统计与分布分析。"""
    import pandas as pd

    from .graph.node import NodeContext
    from .nodes.builtin import analyze as _analyze

    path = Path(args["path"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    ctx = NodeContext(node_id="mcp_analyze", run_id="mcp")
    return _analyze({"data": df}, {"bins": int(args.get("bins", 20))}, ctx)


def tool_fetch_artifact(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """取回某个产物。DataFrame 只回前 N 行。"""
    ref = ArtifactRef(hash=args["hash"], kind=args.get("kind", "json"))
    if not ws.store.exists(ref):
        raise FileNotFoundError(f"产物不存在：{ref}")
    value = ws.store.get(ref)
    if ref.kind == "dataframe":
        limit = int(args.get("limit", 50))
        head = value.head(limit)
        return {
            "rows": int(len(value)),
            "columns": [str(c) for c in value.columns],
            "preview": head.astype(object).where(head.notna(), None).to_dict("records"),
            "truncated": len(value) > limit,
            "notice": SYNTHETIC_NOTICE,
        }
    return {"value": value}


def tool_export(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """把数据集产物写到磁盘，连同数据卡一起。"""
    ref = ArtifactRef(hash=args["hash"], kind="dataframe")
    if not ws.store.exists(ref):
        raise FileNotFoundError(f"产物不存在：{ref}")
    df = ws.store.get(ref)
    out = Path(args["out_path"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")

    # 数据卡与数据同去同回：只给数据不给卡，下游就无从判断它是合成的
    card = {
        "synthetic": True,
        "generated_by": "DataForge (MCP)",
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "artifact": ref.hash,
        "notice": SYNTHETIC_NOTICE,
    }
    card_path = out.with_suffix(out.suffix + ".datacard.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(out), "datacard": str(card_path), "rows": int(len(df)),
            "notice": SYNTHETIC_NOTICE}


def tool_list_runs(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    return {"runs": ws.runstore.runs(limit=int(args.get("limit", 20)))}


def tool_lineage(ws: Workspace, args: dict[str, Any]) -> dict[str, Any]:
    """取一次运行的血缘 —— 「这一列是怎么来的」。"""
    p = ws.root / "lineage" / f"{args['run_id']}.json"
    if not p.exists():
        raise FileNotFoundError(f"没有血缘记录：{args['run_id']}")
    return json.loads(p.read_text(encoding="utf-8"))


# ── 工具注册表 ────────────────────────────────────────────

_FIELD_SCHEMA = {
    "type": "object",
    "required": ["name", "type"],
    "properties": {
        "name": {"type": "string", "description": "蛇形小写英文标识符"},
        "type": {"type": "string",
                 "enum": ["integer", "float", "string", "boolean", "datetime", "categorical"]},
        "description": {"type": "string"},
        "unit": {"type": "string"},
        "min": {"type": "number"},
        "max": {"type": "number"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "start": {"type": "string"}, "end": {"type": "string"},
    },
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "df_plan",
        "description": (
            "把自然语言需求规划成一份数据集计划书。只规划不执行；"
            "返回的 runnable_args 检查无误后可直接传给 df_generate。"
        ),
        "handler": tool_plan,
        "schema": {
            "type": "object", "required": ["text"],
            "properties": {
                "text": {"type": "string", "description": "自然语言需求"},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_generate",
        "description": (
            "按字段定义生成合成数据集，自动过质量门禁（完整性、泄漏检测等）。"
            "产出带 synthetic 标记与完整血缘。注意：门禁阈值不接受外部指定。"
        ),
        "handler": tool_generate,
        "schema": {
            "type": "object", "required": ["fields"],
            "properties": {
                "fields": {"type": "array", "items": _FIELD_SCHEMA},
                "kind": {"type": "string", "enum": ["tabular", "timeseries"]},
                "name": {"type": "string"},
                "num_records": {"type": "integer", "minimum": 1, "maximum": 2000000},
                "target": {"type": "string"},
                "task_type": {"type": "string"},
                "time_column": {"type": "string", "description": "kind=timeseries 时必填"},
                "missing_rate": {"type": "number", "minimum": 0, "maximum": 0.95},
                "noise_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "known_correlated": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "已知合理强相关的字段对，避免被误判为目标泄漏",
                },
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_parse_paper",
        "description": (
            "解析论文 PDF 并抽取数据变量卡（含论文原文出处与置信度）。"
            "只到变量卡为止——这是人工确认闸门，要生成数据需再调 df_dataset_from_paper。"
        ),
        "handler": tool_parse_paper,
        "schema": {
            "type": "object", "required": ["path"],
            "properties": {"path": {"type": "string"}, "name": {"type": "string"}},
        },
    },
    {
        "name": "df_dataset_from_paper",
        "description": (
            "在确认变量之后由论文产出数据集。必须显式列出 adopt——"
            "先看过 df_parse_paper 的变量卡与原文出处再决定。"
        ),
        "handler": tool_dataset_from_paper,
        "schema": {
            "type": "object", "required": ["path", "adopt"],
            "properties": {
                "path": {"type": "string"},
                "adopt": {"type": "array", "items": {"type": "string"},
                          "description": "采纳的变量名，必填且不能为空"},
                "overrides": {"type": "object", "description": "对某个变量的修正，如单位"},
                "answers": {"type": "object", "description": "对 open_questions 的回答"},
                "name": {"type": "string"},
                "num_records": {"type": "integer"},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_rl_dataset",
        "description": (
            "生成强化学习轨迹 (s,a,r,s',done)。门禁会重放环境动力学逐条核对转移——"
            "s' 对不上的数据训出来的是噪声，而统计指标看不出来。"
        ),
        "handler": tool_rl_dataset,
        "schema": {
            "type": "object",
            "properties": {
                "env": {"type": "string", "enum": ["gridworld", "inventory"]},
                "env_params": {"type": "object"},
                "num_episodes": {"type": "integer", "minimum": 1, "maximum": 100000},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 10000},
                "epsilon": {"type": "number", "minimum": 0, "maximum": 1,
                            "description": "探索率。全贪心的轨迹里没有失败样本，离线 RL 学不出「什么不该做」"},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_agent_eval_set",
        "description": (
            "生成 Agent 测评集：任务 + 工具集 + 机器可判的断言。"
            "门禁会用空结果试跑，剔除「什么都不做也能通过」的无区分力判据。"
        ),
        "handler": tool_agent_eval_set,
        "schema": {
            "type": "object",
            "properties": {
                "num_tasks": {"type": "integer", "minimum": 1, "maximum": 5000},
                "categories": {"type": "array", "items": {"type": "string"},
                               "description": "file_ops / compute / safety / structured_output"},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_eval_discrimination",
        "description": (
            "用能力档位不同的模拟 Agent 跑测评集，测这套题有没有区分度。"
            "五个档位通过率一样 = 题白出了。注意：测的是题，不是 Agent。"
        ),
        "handler": tool_eval_discrimination,
        "schema": {
            "type": "object",
            "properties": {
                "num_tasks": {"type": "integer", "minimum": 5, "maximum": 2000},
                "categories": {"type": "array", "items": {"type": "string"}},
                "difficulties": {"type": "array", "items": {"type": "string",
                                 "enum": ["easy", "medium", "hard"]}},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_export_inspect",
        "description": (
            "把测评集导出成 Inspect（UK AISI）格式，用真实模型跑。"
            "判对错的仍是 DataForge 的断言，不是 model-graded scorer。需要 Docker。"
        ),
        "handler": tool_export_inspect,
        "schema": {
            "type": "object", "required": ["out_dir"],
            "properties": {
                "out_dir": {"type": "string"},
                "num_tasks": {"type": "integer", "minimum": 1, "maximum": 5000},
                "categories": {"type": "array", "items": {"type": "string"}},
                "difficulties": {"type": "array", "items": {"type": "string"}},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_run_assertions",
        "description": (
            "用测评任务的断言判定一次 Agent 运行的成败。判对错的是代码，不是模型。"
        ),
        "handler": tool_run_assertions,
        "schema": {
            "type": "object", "required": ["task", "result"],
            "properties": {
                "task": {"type": "object", "description": "含 assertions 的测评任务"},
                "result": {"type": "object",
                           "description": "{output, tool_calls:[{name}], error}"},
                "workdir": {"type": "string", "description": "文件类断言的根目录"},
            },
        },
    },
    {
        "name": "df_llm_dataset",
        "description": (
            "生成 SFT 指令对与 DPO 偏好对，自带去重与**基准污染检查**。"
            "训练集混进评测题目会让分数虚高——那看起来像「效果很好」。"
        ),
        "handler": tool_llm_dataset,
        "schema": {
            "type": "object",
            "properties": {
                "num_samples": {"type": "integer", "minimum": 1, "maximum": 100000},
                "with_dpo": {"type": "boolean"},
                "reference": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "[基准名, 题面] 列表，用于污染检查。不传则用内置样例",
                },
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_inspect_source",
        "description": (
            "看一个外部链接（HF 数据集 / GitHub 仓库 / 已知基准名）对应什么字段结构。"
            "只读元数据，不下载数据本体，也不生成任何数据。"
        ),
        "handler": tool_inspect_source,
        "schema": {
            "type": "object", "required": ["link"],
            "properties": {
                "link": {"type": "string",
                         "description": "hf://datasets/x/y、github.com/org/repo 或基准名"},
                "allow_network": {"type": "boolean",
                                  "description": "默认 false，只查本地基准注册表"},
            },
        },
    },
    {
        "name": "df_align_to_source",
        "description": (
            "按外部链接的结构生成**同构合成数据**。"
            "注意：产出与原数据集内容无关，不得用于刷榜或充当真实基准评测结果。"
        ),
        "handler": tool_align_to_source,
        "schema": {
            "type": "object", "required": ["link"],
            "properties": {
                "link": {"type": "string"},
                "num_records": {"type": "integer", "minimum": 1, "maximum": 2000000},
                "name": {"type": "string"},
                "target": {"type": "string"},
                "allow_network": {"type": "boolean"},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_list_benchmarks",
        "description": "列出本地已知的基准及其结构与评测协议。",
        "handler": tool_list_benchmarks,
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "df_validate",
        "description": (
            "对已有数据集跑质量与泄漏检测（目标泄漏 / 切分重叠 / 时间穿越）。"
            "支持 csv / parquet / json / jsonl。"
        ),
        "handler": tool_validate,
        "schema": {
            "type": "object", "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "target": {"type": "string", "description": "标签列，用于目标泄漏检测"},
                "known_correlated": {"type": "array",
                                     "items": {"type": "array", "items": {"type": "string"}}},
            },
        },
    },
    {
        "name": "df_analyze",
        "description": "对已有数据集做描述统计、分布直方图与相关性分析。",
        "handler": tool_analyze,
        "schema": {
            "type": "object", "required": ["path"],
            "properties": {"path": {"type": "string"}, "bins": {"type": "integer"}},
        },
    },
    {
        "name": "df_fetch_artifact",
        "description": "取回某个产物的内容。DataFrame 只返回前 N 行预览。",
        "handler": tool_fetch_artifact,
        "schema": {
            "type": "object", "required": ["hash"],
            "properties": {
                "hash": {"type": "string"},
                "kind": {"type": "string", "enum": ["dataframe", "json", "text"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "df_export",
        "description": "把数据集写到磁盘，同时生成数据卡（synthetic 标记不可关闭）。",
        "handler": tool_export,
        "schema": {
            "type": "object", "required": ["hash", "out_path"],
            "properties": {"hash": {"type": "string"}, "out_path": {"type": "string"}},
        },
    },
    {
        "name": "df_list_runs",
        "description": "列出历史运行。",
        "handler": tool_list_runs,
        "schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    {
        "name": "df_lineage",
        "description": "取一次运行的血缘：某一列是怎么来的（节点、参数、人工决策、论文出处）。",
        "handler": tool_lineage,
        "schema": {
            "type": "object", "required": ["run_id"],
            "properties": {"run_id": {"type": "string"}},
        },
    },
]

BY_NAME = {t["name"]: t for t in TOOLS}


def call_tool(ws: Workspace, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """按名字调用工具。协议无关，方便直接单测。"""
    spec = BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"未知工具 {name!r}，可用：{sorted(BY_NAME)}")
    return spec["handler"](ws, args or {})


# ── MCP 协议绑定 ──────────────────────────────────────────

def build_server(ws: Workspace):  # pragma: no cover - 需要 mcp 运行时
    """绑定到 MCP 协议。

    MCP 2.0 用构造函数回调（``on_list_tools`` / ``on_call_tool``），
    不再是 1.x 那套 ``@server.list_tools()`` 装饰器。
    """
    import mcp.types as types
    from mcp.server import Server

    from . import __version__

    async def on_list_tools(ctx: Any, params: Any) -> "types.ListToolsResult":
        return types.ListToolsResult(tools=[
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["schema"],
            )
            for t in TOOLS
        ])

    async def on_call_tool(ctx: Any, params: Any) -> "types.CallToolResult":
        try:
            result: dict[str, Any] = call_tool(ws, params.name, params.arguments or {})
            is_error = False
        except Exception as e:  # noqa: BLE001 - 工具出错要变成可读回复，不能炸掉连接
            result = {"error": f"{type(e).__name__}: {e}"}
            is_error = True
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2, default=str),
            )],
            isError=is_error,
        )

    return Server(
        "dataforge",
        version=__version__,
        instructions=(
            "DataForge 生成可复现的合成数据集，自带质量门禁与泄漏检测。"
            "所有产出都带 synthetic 标记与完整血缘，不得用于刷榜或充当真实评测结果。"
            "论文链路有人工确认闸门：df_parse_paper 只给变量卡，"
            "要生成数据必须再调 df_dataset_from_paper 并显式列出采纳的变量。"
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - 入口
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(prog="dataforge-mcp", description="DataForge MCP 服务端")
    ap.add_argument("--workspace", default="./.dataforge")
    args = ap.parse_args(argv)

    ws = Workspace(args.workspace)

    async def run() -> None:
        from mcp.server.stdio import stdio_server

        server = build_server(ws)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        asyncio.run(run())
    finally:
        ws.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
