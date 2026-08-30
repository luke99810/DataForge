"""内置节点 —— L2 能力层对 L1 内核的桥接。

这些函数是**纯的**：拿到已取回的输入，返回一个值。它们不碰制品库、
不知道缓存存在，所以可以脱离执行器单测。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dataset_generator.core.config import DatasetConfig
from dataset_generator.core.schema import DatasetSchema, FieldSchema
from dataset_generator.core.types import Domain, FieldType, OutputFormat, TaskType
from dataset_generator.generators.tabular import TabularGenerator
from dataset_generator.templates import get_domain_templates
from dataset_generator.validators.quality import DataQualityValidator

from ..graph.node import NodeContext
from ..harness.gates import (
    GateConfig, GateReport, gate_completeness, gate_leakage, gate_row_count, gate_schema,
)

# 每个节点函数各自的版本号。改了实现必须升 —— 见设计文档 §5.2。
V = {
    "schema_from_domain": "1.0.0",
    "schema_from_fields": "1.0.0",
    "generate_tabular": "1.0.0",
    "split_chrono": "1.0.0",
    "split_random": "1.0.0",
    "quality_gate": "1.0.0",
    "analyze": "1.0.0",
    "export_bundle": "1.0.0",
}


# ── SCHEMA ────────────────────────────────────────────────

def schema_from_domain(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """按领域模板生成 Schema（序列化成 dict 便于进制品库）。"""
    domain = Domain(params["domain"])
    tpl = get_domain_templates(domain)
    # 模板 field dict 的形状与 FieldSchema.from_dict 一致（含 constraints），
    # 直接复用内核的反序列化，避免手工映射漏掉 constraints
    fields = [FieldSchema.from_dict(f) for f in tpl["fields"]]
    schema = DatasetSchema(
        name=params.get("name", f"{domain.value}_dataset"),
        description=tpl.get("description", ""),
        domain=domain.value,
        fields=fields,
        target_field=params.get("target_field"),
    )
    errs = schema.validate()
    if errs:
        raise ValueError(f"Schema 非法：{errs}")
    for f in fields:
        ctx.column_source(f.name, origin="domain_template", domain=domain.value)
    ctx.note(f"由领域模板 {domain.value} 生成 {len(fields)} 个字段")
    return schema.to_dict()


def schema_from_fields(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """由显式字段定义构建 Schema。

    params["fields"] 形如
    ``[{"name": "speed", "type": "float", "min": 0, "max": 70, "unit": "mph"}]``
    """
    fields = []
    for f in params["fields"]:
        fs = FieldSchema(
            name=f["name"],
            field_type=FieldType(f["type"]),
            description=f.get("description", ""),
            nullable=f.get("nullable", False),
        )
        for k in ("min", "max", "categories", "weights", "precision", "start", "end"):
            if k in f:
                setattr(fs.constraints, k, f[k])
        fields.append(fs)
        ctx.column_source(
            f["name"],
            origin=f.get("origin", "manual"),
            unit=f.get("unit"),
            source_span=f.get("source_span"),
        )
    schema = DatasetSchema(
        name=params.get("name", "dataset"),
        description=params.get("description", ""),
        domain=params.get("domain", "custom"),
        fields=fields,
        target_field=params.get("target_field"),
    )
    errs = schema.validate()
    if errs:
        raise ValueError(f"Schema 非法：{errs}")
    return schema.to_dict()


# ── GENERATE ──────────────────────────────────────────────

def generate_tabular(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> pd.DataFrame:
    """按 Schema 生成表格数据。seed 来自执行期上下文，保证可复现。"""
    schema = DatasetSchema.from_dict(inputs["schema"])
    cfg = DatasetConfig(
        project_name=params.get("project_name", schema.name),
        domain=Domain(schema.domain) if schema.domain else Domain.CUSTOM,
        task_type=TaskType(params.get("task_type", "classification")),
        num_records=int(params["num_records"]),
        output_format=OutputFormat.CSV,
        seed=int(params.get("seed", ctx.seed if ctx.seed is not None else 42)),
        noise_ratio=float(params.get("noise_ratio", 0.0)),
        missing_rate=float(params.get("missing_rate", 0.0)),
    )
    df = TabularGenerator(cfg, schema).generate()
    ctx.note(f"生成 {len(df)} 行 × {len(df.columns)} 列", seed=cfg.seed)
    return df


# ── TRANSFORM ─────────────────────────────────────────────

def split_chrono(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """按时间顺序切分。

    时序任务必须用这个，不能用随机切分 —— 随机切分产生的虚高精度
    极具迷惑性（设计文档 §8）。
    """
    df: pd.DataFrame = inputs["data"]
    col = params["time_column"]
    if col not in df.columns:
        raise ValueError(f"时间列 {col!r} 不存在，现有列：{list(df.columns)}")
    ratios = params.get("ratios", [0.7, 0.1, 0.2])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"切分比例之和必须为 1，得到 {sum(ratios)}")

    ordered = df.sort_values(col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    out = {
        "train": ordered.iloc[:i1].reset_index(drop=True),
        "val": ordered.iloc[i1:i2].reset_index(drop=True),
        "test": ordered.iloc[i2:].reset_index(drop=True),
    }
    ctx.note("按时间切分", sizes={k: len(v) for k, v in out.items()}, time_column=col)
    return {k: v.to_dict("list") for k, v in out.items()}


def split_random(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """随机切分。**时序任务禁用** —— 这里显式拦下，而不是留给门禁警告。"""
    if params.get("task_type") in ("time_series_forecast",):
        raise ValueError(
            "时序任务禁止随机切分：它几乎总是错的，且产生的虚高精度极具迷惑性。"
            "请改用 split_chrono。"
        )
    df: pd.DataFrame = inputs["data"]
    ratios = params.get("ratios", [0.7, 0.1, 0.2])
    rng = np.random.default_rng(int(params.get("seed", ctx.seed or 42)))
    idx = rng.permutation(len(df))
    n = len(df)
    i1, i2 = int(n * ratios[0]), int(n * ratios[0]) + int(n * ratios[1])
    parts = {"train": idx[:i1], "val": idx[i1:i2], "test": idx[i2:]}
    return {k: df.iloc[v].reset_index(drop=True).to_dict("list") for k, v in parts.items()}


# ── GATE ──────────────────────────────────────────────────

def quality_gate(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """质量门禁 —— 唯一可阻断的节点类型。

    不通过就抛异常，不返回「有问题但先往下走」的结果。
    """
    df: pd.DataFrame = inputs["data"]
    schema = DatasetSchema.from_dict(inputs["schema"]) if "schema" in inputs else None

    cfg = GateConfig()
    for k, v in (params.get("thresholds") or {}).items():
        cfg.set(k, v, reason=params.get("threshold_reason", ""))

    report = GateReport(config=cfg)
    report.add(gate_row_count(df, cfg))
    if schema is not None:
        report.add(gate_schema(df, schema))
    report.add(gate_completeness(df, cfg))

    splits = None
    if "splits" in inputs:
        splits = {k: pd.DataFrame(v) for k, v in inputs["splits"].items()}
    report.add(gate_leakage(
        df,
        target=params.get("target"),
        splits=splits,
        time_column=params.get("time_column"),
        known_correlated=tuple(tuple(p) for p in params.get("known_correlated", [])),
    ))

    # 内核自带的四维校验也并进来，作为 WARN 记录
    if schema is not None:
        issues = DataQualityValidator(schema).validate(df)
        errs = [i for i in issues if i.severity == "error"]
        if errs:
            from ..harness.gates import GateResult, Level
            report.add(GateResult(
                "kernel_validator", Level.WARN, False,
                f"内核校验发现 {len(errs)} 个 error 级问题",
                {"issues": [{"severity": i.severity, "category": i.category,
                             "message": i.message, "column": i.column} for i in issues]},
            ))

    if cfg.relaxed_any:
        ctx.note("门禁阈值被放宽", audit=cfg.audit())
        ctx.decision("放宽门禁阈值", cfg.overrides, by="config",
                     note="已记入运行日志与数据字典")

    report.raise_if_needed()
    return report.to_dict()


# ── ANALYZE ───────────────────────────────────────────────

def analyze(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """描述统计 + 缺失模式 + 相关性，供界面画图。"""
    df: pd.DataFrame = inputs["data"]
    num = df.select_dtypes(include=[np.number])

    desc: dict[str, Any] = {}
    for c in num.columns:
        s = num[c].dropna()
        if s.empty:
            continue
        desc[str(c)] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    corr: dict[str, dict[str, float]] = {}
    if len(num.columns) > 1:
        cm = num.corr()
        corr = {
            str(a): {str(b): (None if pd.isna(cm.loc[a, b]) else float(cm.loc[a, b]))
                     for b in cm.columns}
            for a in cm.index
        }

    hist: dict[str, Any] = {}
    for c in list(num.columns)[: int(params.get("max_hist_cols", 12))]:
        s = num[c].dropna()
        if s.empty or s.nunique() < 2:
            continue
        counts, edges = np.histogram(s, bins=int(params.get("bins", 20)))
        hist[str(c)] = {"counts": counts.tolist(), "edges": edges.tolist()}

    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "describe": desc,
        "correlation": corr,
        "histograms": hist,
        "missing": {str(c): float(df[c].isna().mean()) for c in df.columns},
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
    }


# ── EXPORT ────────────────────────────────────────────────

def export_bundle(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """导出制品清单 + 数据字典。

    强制带 ``synthetic: true`` —— 设计文档 §12 红线一，不提供关闭选项。
    """
    df: pd.DataFrame = inputs["data"]
    schema = DatasetSchema.from_dict(inputs["schema"]) if "schema" in inputs else None
    gate = inputs.get("gate", {})

    card = {
        "synthetic": True,  # 不可关闭：合成数据不得冒充真实评测
        "name": params.get("name", "dataset"),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "generated_by": "DataForge",
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "gate_audit": gate.get("audit", {}),
        "notice": (
            "本数据集为合成数据。不得用于刷榜，也不得作为真实评测结果报告。"
        ),
    }
    if schema is not None:
        card["schema"] = schema.to_dict()
    return card
