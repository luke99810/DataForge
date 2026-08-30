"""预置流水线 —— 把内置节点串成可直接跑的 DAG。

这些是 L3 规划器最终要生成的东西的手写版本。先把形状固定下来，
LLM 规划器接进来时产出同样结构的 DAG 即可。
"""

from __future__ import annotations

from typing import Any, Sequence

from .graph.dag import DAG
from .graph.node import NodeSpec
from .nodes import builtin as B
from .nodes import agenteval as A
from .nodes import llmdata as L
from .nodes import paper as P
from .nodes import rl as R
from .nodes import source as SRC


def tabular_pipeline(
    *,
    fields: Sequence[dict[str, Any]],
    num_records: int = 1000,
    name: str = "dataset",
    target: str | None = None,
    task_type: str = "classification",
    missing_rate: float = 0.0,
    noise_ratio: float = 0.0,
    thresholds: dict[str, float] | None = None,
    known_correlated: Sequence[Sequence[str]] = (),
) -> DAG:
    """由显式字段定义生成一份表格数据，并过质量门禁。

    形状::

        schema ──> generate ──┬──> gate ──> export
                              └──> analyze
    """
    return DAG([
        NodeSpec(
            "schema", "SCHEMA", B.schema_from_fields,
            params={"fields": list(fields), "name": name, "target_field": target},
            code_version=B.V["schema_from_fields"],
            description="由字段定义构建 Schema",
        ),
        NodeSpec(
            "generate", "GENERATE", B.generate_tabular,
            inputs={"schema": "schema"},
            params={
                "num_records": num_records,
                "task_type": task_type,
                "missing_rate": missing_rate,
                "noise_ratio": noise_ratio,
            },
            code_version=B.V["generate_tabular"],
            description="按 Schema 生成表格数据",
        ),
        NodeSpec(
            "analyze", "ANALYZE", B.analyze,
            inputs={"data": "generate"},
            code_version=B.V["analyze"],
            description="统计与分布",
        ),
        NodeSpec(
            "gate", "GATE", B.quality_gate,
            inputs={"data": "generate", "schema": "schema"},
            params={
                "target": target,
                "thresholds": thresholds or {},
                "known_correlated": [list(p) for p in known_correlated],
            },
            code_version=B.V["quality_gate"],
            description="质量门禁（可阻断）",
        ),
        NodeSpec(
            "export", "EXPORT", B.export_bundle,
            inputs={"data": "generate", "schema": "schema", "gate": "gate"},
            params={"name": name},
            code_version=B.V["export_bundle"],
            description="导出数据卡与制品清单",
        ),
    ])


def timeseries_pipeline(
    *,
    fields: Sequence[dict[str, Any]],
    time_column: str,
    num_records: int = 2000,
    name: str = "timeseries",
    target: str | None = None,
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
    thresholds: dict[str, float] | None = None,
    known_correlated: Sequence[Sequence[str]] = (),
) -> DAG:
    """时序流水线 —— 强制按时间切分。

    ``split_chrono`` 是硬编码的，不给随机切分的选项：
    时序任务的随机切分几乎总是错的，而它产生的虚高精度极具迷惑性。
    """
    return DAG([
        NodeSpec(
            "schema", "SCHEMA", B.schema_from_fields,
            params={"fields": list(fields), "name": name, "target_field": target},
            code_version=B.V["schema_from_fields"],
        ),
        NodeSpec(
            "generate", "GENERATE", B.generate_tabular,
            inputs={"schema": "schema"},
            params={"num_records": num_records, "task_type": "time_series_forecast"},
            code_version=B.V["generate_tabular"],
        ),
        NodeSpec(
            "split", "TRANSFORM", B.split_chrono,
            inputs={"data": "generate"},
            params={"time_column": time_column, "ratios": list(ratios)},
            code_version=B.V["split_chrono"],
            description="按时间顺序切分（禁止随机切分）",
        ),
        NodeSpec(
            "analyze", "ANALYZE", B.analyze,
            inputs={"data": "generate"},
            code_version=B.V["analyze"],
        ),
        NodeSpec(
            "gate", "GATE", B.quality_gate,
            inputs={"data": "generate", "schema": "schema", "splits": "split"},
            params={
                "target": target,
                "time_column": time_column,
                "thresholds": thresholds or {},
                "known_correlated": [list(p) for p in known_correlated],
            },
            code_version=B.V["quality_gate"],
        ),
        NodeSpec(
            "export", "EXPORT", B.export_bundle,
            inputs={"data": "generate", "schema": "schema", "gate": "gate"},
            params={"name": name},
            code_version=B.V["export_bundle"],
        ),
    ])


def paper_pipeline(
    *,
    pdf_path: str,
    name: str = "paper_dataset",
    num_records: int = 1000,
    decisions: dict[str, Any] | None = None,
    max_chars: int = 14_000,
    min_confidence: float = 0.5,
    thresholds: dict[str, float] | None = None,
) -> DAG:
    """论文驱动的数据集流水线。

    形状::

        parse ──> extract ──> confirm ──> schema ──> generate ──┬──> gate ──> export
             (规则)    (LLM)    (★人工)    (代码)                └──> analyze

    ``decisions`` 为空时会停在 ``confirm``；界面回填后把答案作为该节点参数
    重新提交——上游全部命中缓存，只有 confirm 及其下游重算。
    **续跑不需要单独的恢复逻辑，它是指纹缓存的自然结果。**
    """
    return DAG([
        NodeSpec(
            "parse", "SOURCE", P.parse_pdf,
            params={"path": pdf_path},
            code_version=P.V["parse_pdf"],
            description="解析 PDF 并切分章节",
        ),
        NodeSpec(
            "extract", "EXTRACT", P.extract_variables,
            inputs={"paper": "parse"},
            params={"max_chars": max_chars, "min_confidence": min_confidence},
            code_version=P.V["extract_variables"],
            description="模型抽取变量卡（唯一调用 LLM 的节点）",
        ),
        NodeSpec(
            "confirm", "GATE", P.confirm_variables,
            inputs={"extraction": "extract"},
            params={"decisions": decisions or {}},
            code_version=P.V["confirm_variables"],
            description="人工确认闸门（不可跳过）",
        ),
        NodeSpec(
            "schema", "SCHEMA", P.schema_from_cards,
            inputs={"confirmed": "confirm"},
            params={"name": name},
            code_version=P.V["schema_from_cards"],
            description="由确认变量构建 Schema",
        ),
        NodeSpec(
            "generate", "GENERATE", B.generate_tabular,
            inputs={"schema": "schema"},
            params={"num_records": num_records},
            code_version=B.V["generate_tabular"],
        ),
        NodeSpec(
            "analyze", "ANALYZE", B.analyze,
            inputs={"data": "generate"},
            code_version=B.V["analyze"],
        ),
        NodeSpec(
            "gate", "GATE", B.quality_gate,
            inputs={"data": "generate", "schema": "schema"},
            params={"thresholds": thresholds or {}},
            code_version=B.V["quality_gate"],
        ),
        NodeSpec(
            "export", "EXPORT", B.export_bundle,
            inputs={"data": "generate", "schema": "schema", "gate": "gate"},
            params={"name": name},
            code_version=B.V["export_bundle"],
        ),
    ])


def rl_pipeline(
    *,
    env: str = "gridworld",
    env_params: dict[str, Any] | None = None,
    num_episodes: int = 200,
    max_steps: int = 50,
    epsilon: float = 0.2,
    name: str = "rl_dataset",
) -> DAG:
    """RL 轨迹流水线。

    ``trajectory_gate`` 会**重放动力学**核对每条转移——
    s' 不是从 (s,a) 走出来的数据，训出来的模型学的是噪声，
    而这件事从统计指标上完全看不出来。
    """
    return DAG([
        NodeSpec(
            "generate", "GENERATE", R.generate_trajectories,
            params={"env": env, "env_params": env_params or {},
                    "num_episodes": num_episodes, "max_steps": max_steps,
                    "epsilon": epsilon},
            code_version=R.V["generate_trajectories"],
            description="生成 (s,a,r,s',done) 轨迹",
        ),
        NodeSpec(
            "gate", "GATE", R.trajectory_gate,
            inputs={"trajectories": "generate"},
            code_version=R.V["trajectory_gate"],
            description="重放动力学核对转移自洽性",
        ),
    ])


def agent_eval_pipeline(
    *,
    num_tasks: int = 20,
    categories: list[str] | None = None,
    name: str = "agent_eval",
) -> DAG:
    """Agent 测评集流水线。

    ``agent_eval_gate`` 不只校验断言定义，还**用空结果真跑一遍**：
    在「什么都不做」时也能通过的判据，区分不出好坏 Agent。
    """
    return DAG([
        NodeSpec(
            "generate", "GENERATE", A.generate_agent_eval,
            params={"num_tasks": num_tasks, "categories": categories},
            code_version=A.V["generate_agent_eval"],
            description="生成任务 + 工具集 + 机器可判的断言",
        ),
        NodeSpec(
            "gate", "GATE", A.agent_eval_gate,
            inputs={"eval_set": "generate"},
            code_version=A.V["agent_eval_gate"],
            description="断言可执行性与区分力检查",
        ),
    ])


def llm_sft_pipeline(
    *,
    num_samples: int = 200,
    with_dpo: bool = True,
    reference: list | None = None,
    name: str = "llm_dataset",
) -> DAG:
    """LLM 训练集流水线：SFT（+ 可选 DPO）。

    门禁里**基准污染是零容忍**：训练集混进评测题目会让分数虚高，
    而虚高的分数看起来像「效果很好」，比报错危险得多。
    """
    nodes = [
        NodeSpec(
            "sft", "GENERATE", L.generate_sft,
            params={"num_samples": num_samples},
            code_version=L.V["generate_sft"],
            description="生成 SFT 指令对并去重",
        ),
        NodeSpec(
            "gate", "GATE", L.llmdata_gate,
            inputs={"dataset": "sft"},
            params={"reference": reference},
            code_version=L.V["llmdata_gate"],
            description="重复率 + 基准污染检查",
        ),
    ]
    if with_dpo:
        nodes += [
            NodeSpec(
                "dpo", "GENERATE", L.generate_dpo,
                inputs={"sft": "sft"},
                code_version=L.V["generate_dpo"],
                description="按明确规则劣化出偏好对",
            ),
            NodeSpec(
                "dpo_gate", "GATE", L.llmdata_gate,
                inputs={"dataset": "dpo"},
                params={"reference": reference},
                code_version=L.V["llmdata_gate"],
                description="偏好对有效性检查",
            ),
        ]
    return DAG(nodes)


def _fields_from_source(inputs: dict[str, Any], params: dict[str, Any], ctx: Any) -> dict:
    """把 resolve_source 的产物转成 schema_from_fields 认识的形状。"""
    from dataset_generator.core.schema import DatasetSchema, FieldSchema
    from dataset_generator.core.types import FieldType

    src = inputs["source"]
    fields = []
    for f in src["fields"]:
        fs = FieldSchema(
            name=f["name"],
            field_type=FieldType(f["type"]),
            description=f.get("description", ""),
        )
        for k in ("min", "max", "categories"):
            if f.get(k) is not None:
                setattr(fs.constraints, k, f[k])
        fields.append(fs)
        ctx.column_source(f["name"], origin=src["resolved_by"],
                          **{k: v for k, v in src["provenance"].items() if k != "note"})

    schema = DatasetSchema(
        name=params.get("name", "aligned_dataset"),
        description=(src.get("benchmark") or {}).get("task", ""),
        domain="custom",
        fields=fields,
        target_field=params.get("target"),
    )
    errs = schema.validate()
    if errs:
        raise ValueError(f"由外部结构构建的 Schema 非法：{errs}")
    ctx.note(f"由 {src['resolved_by']} 的结构构建 Schema，{len(fields)} 个字段")
    return schema.to_dict()


def _export_aligned(inputs: dict[str, Any], params: dict[str, Any], ctx: Any) -> dict:
    """导出，并把来源与合成声明一起写进数据卡。

    这里的 ``synthetic`` 与 ``notice`` **不提供关闭选项**：
    同构合成数据被误当成真实基准结果，是设计文档 §12 的红线一。
    """
    df = inputs["data"]
    src = inputs["source"]
    gate = inputs.get("gate", {})
    bench = src.get("benchmark") or {}
    return {
        "synthetic": True,
        "name": params.get("name", "aligned_dataset"),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "generated_by": "DataForge",
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "aligned_to": {
            "resolved_by": src["resolved_by"],
            "ref": src["ref"],
            "benchmark": bench.get("name"),
            "protocol": src.get("protocol"),
            "skipped_fields": [s["name"] for s in src.get("skipped", [])],
        },
        "provenance": src["provenance"],
        "gate_audit": gate.get("audit", {}),
        "notice": src.get("notice"),
    }


def source_aligned_pipeline(
    *,
    link: str,
    num_records: int = 1000,
    name: str = "aligned_dataset",
    target: str | None = None,
    allow_network: bool = False,
    thresholds: dict[str, float] | None = None,
) -> DAG:
    """由外部链接生成**同构**合成数据集。

    形状::

        source ──> schema ──> generate ──┬──> gate ──> export
                                          └──> analyze

    ``allow_network=False``（默认）时只查本地基准注册表。
    **出网是显式选择**——默认不出，省得一次误操作就把请求发出去。
    """
    return DAG([
        NodeSpec(
            "source", "SOURCE", SRC.resolve_source,
            params={"link": link, "allow_network": allow_network},
            code_version=SRC.V["resolve_source"],
            description="解析外部链接的结构（只取元数据）",
        ),
        NodeSpec(
            "schema", "SCHEMA", _fields_from_source,
            inputs={"source": "source"},
            params={"name": name, "target": target},
            code_version="1.0.0",
            description="由外部结构构建 Schema",
        ),
        NodeSpec(
            "generate", "GENERATE", B.generate_tabular,
            inputs={"schema": "schema"},
            params={"num_records": num_records},
            code_version=B.V["generate_tabular"],
        ),
        NodeSpec(
            "analyze", "ANALYZE", B.analyze,
            inputs={"data": "generate"},
            code_version=B.V["analyze"],
        ),
        NodeSpec(
            "gate", "GATE", B.quality_gate,
            inputs={"data": "generate", "schema": "schema"},
            params={"thresholds": thresholds or {}},
            code_version=B.V["quality_gate"],
        ),
        NodeSpec(
            "export", "EXPORT", _export_aligned,
            inputs={"data": "generate", "source": "source", "gate": "gate"},
            params={"name": name},
            code_version="1.0.0",
            description="导出并写入来源与合成声明",
        ),
    ])
