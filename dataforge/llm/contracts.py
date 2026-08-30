"""工具契约 —— 模型输出的 JSON Schema 与校验。

设计文档 §3.2：**输入和输出都校验**。只校验输入是不够的，
模型看不到自己输出的结构错误，会把脏数据当成功继续往下传。

这里手写了一个够用的校验器，不引入 jsonschema：
需要的只是 type / required / enum / 数值区间 / 数组元素这几样，
而桌面版要打包，少一个依赖少一分打包风险。
"""

from __future__ import annotations

import re
from typing import Any

VALID_DTYPES = ("integer", "float", "string", "boolean", "datetime", "categorical")
VALID_ROLES = ("input", "target", "control", "structural")

# 合法字段名：ASCII 标识符。**不要用 isalnum()**，见 check_variable_semantics
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ContractError(ValueError):
    """模型输出不符合契约。"""


def _fail(path: str, msg: str) -> None:
    raise ContractError(f"{path or '(根)'}: {msg}")


def validate_against(obj: Any, schema: dict[str, Any], path: str = "") -> None:
    """按 schema 校验 obj。不合规抛 :class:`ContractError`，错误信息要能直接喂回模型。"""
    t = schema.get("type")

    if t == "object":
        if not isinstance(obj, dict):
            _fail(path, f"应为对象，实际是 {type(obj).__name__}")
        for k in schema.get("required", []):
            if k not in obj or obj[k] is None:
                _fail(path, f"缺少必填字段 {k!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(obj) - set(props))
            if extra:
                _fail(path, f"出现未定义字段 {extra}")
        for k, sub in props.items():
            if k in obj and obj[k] is not None:
                validate_against(obj[k], sub, f"{path}.{k}" if path else k)
        return

    if t == "array":
        if not isinstance(obj, list):
            _fail(path, f"应为数组，实际是 {type(obj).__name__}")
        if "minItems" in schema and len(obj) < schema["minItems"]:
            _fail(path, f"至少需要 {schema['minItems']} 个元素，实际 {len(obj)}")
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            _fail(path, f"最多 {schema['maxItems']} 个元素，实际 {len(obj)}")
        item = schema.get("items")
        if item:
            for i, v in enumerate(obj):
                validate_against(v, item, f"{path}[{i}]")
        return

    if t == "string":
        if not isinstance(obj, str):
            _fail(path, f"应为字符串，实际是 {type(obj).__name__}")
    elif t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            _fail(path, f"应为数字，实际是 {type(obj).__name__}")
    elif t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            _fail(path, f"应为整数，实际是 {type(obj).__name__}")
    elif t == "boolean":
        if not isinstance(obj, bool):
            _fail(path, f"应为布尔值，实际是 {type(obj).__name__}")

    if "enum" in schema and obj not in schema["enum"]:
        _fail(path, f"取值必须是 {schema['enum']} 之一，实际是 {obj!r}")
    if "minimum" in schema and isinstance(obj, (int, float)) and obj < schema["minimum"]:
        _fail(path, f"不得小于 {schema['minimum']}，实际 {obj}")
    if "maximum" in schema and isinstance(obj, (int, float)) and obj > schema["maximum"]:
        _fail(path, f"不得大于 {schema['maximum']}，实际 {obj}")


# ── 变量抽取契约 ──────────────────────────────────────────

SOURCE_SPAN = {
    "type": "object",
    "required": ["page", "quote"],
    "properties": {
        "page": {"type": "integer", "minimum": 1},
        "section": {"type": "string"},
        "quote": {"type": "string"},
    },
}

VARIABLE_CARD = {
    "type": "object",
    "required": ["name", "dtype", "role", "source_span", "confidence"],
    "properties": {
        "name": {"type": "string"},
        "symbol": {"type": "string"},
        "dtype": {"type": "string", "enum": list(VALID_DTYPES)},
        "unit": {"type": "string"},
        "min": {"type": "number"},
        "max": {"type": "number"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "role": {"type": "string", "enum": list(VALID_ROLES)},
        "distribution_hint": {"type": "string"},
        "description": {"type": "string"},
        # 必填：给不出论文原文位置的变量，基本可以断定是模型编的
        "source_span": SOURCE_SPAN,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        # 模型自己标注「这条是不是来自相关工作而非本文」
        "from_related_work": {"type": "boolean"},
    },
}

EXTRACT_VARIABLES = {
    "type": "object",
    "required": ["variables"],
    "properties": {
        "variables": {"type": "array", "items": VARIABLE_CARD, "maxItems": 60},
        "task_type": {
            "type": "string",
            "enum": ["classification", "regression", "time_series_forecast",
                     "clustering", "anomaly_detection", "unknown"],
        },
        "dataset_hint": {
            "type": "object",
            "properties": {
                "num_records": {"type": "integer", "minimum": 1},
                "missing_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "time_granularity": {"type": "string"},
                "notes": {"type": "string"},
            },
        },
        # 论文里没说清、需要人来定的地方
        "open_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "why": {"type": "string"},
                },
            },
        },
    },
}


# ── 校验之外的语义检查 ────────────────────────────────────

def check_variable_semantics(card: dict[str, Any]) -> list[str]:
    """schema 管不到的语义问题。

    返回问题列表（空 = 没问题）。这些不是「格式错」而是「说不通」，
    所以单独一层：格式错要重试，说不通要交给人。
    """
    issues: list[str] = []
    name = card.get("name", "")

    # 注意不能用 str.isalnum()：它是 Unicode 感知的，"道路编号".isalnum() 为 True，
    # 中文字段名会被放过去，到了生成器那里才炸
    if not IDENT.fullmatch(name or ""):
        issues.append(f"字段名 {name!r} 不是合法标识符（需为英文字母/数字/下划线，且不以数字开头）")

    lo, hi = card.get("min"), card.get("max")
    if lo is not None and hi is not None and lo > hi:
        issues.append(f"{name}: min({lo}) 大于 max({hi})")

    if card.get("dtype") == "categorical" and not card.get("categories"):
        issues.append(f"{name}: categorical 类型必须给出 categories")

    quote = (card.get("source_span") or {}).get("quote", "")
    if len(quote.strip()) < 8:
        issues.append(f"{name}: 原文引用太短（{quote!r}），无法回溯")

    return issues


SYSTEM_EXTRACT = """你是一名数据集工程师。从论文正文中抽取**这篇论文自己使用的**数据变量。

硬性要求：
1. 每个变量必须给出 source_span（页码 + 原文引用），引用必须是论文里的原句。
   给不出原文位置的变量不要输出——宁可少，不要编。
2. 只抽取本文实验真正使用的变量。若某变量出现在「相关工作」章节、
   是别人用过而本文没用的，把 from_related_work 设为 true 并把 confidence 压到 0.4 以下。
3. 论文没写清楚的关键参数（如缺失率、单位冲突、切分比例），
   不要替作者猜，写进 open_questions。
4. confidence 反映你的把握：正文明确写出的接近 1.0，靠推断的低于 0.6。
5. 只输出 JSON，不要任何解释文字。"""


# ── 规划契约 ──────────────────────────────────────────────

# 模型产出的是**计划书**，不是图。
#
# 让模型直接输出 DAG 是有诱惑力的，但它能凭空造出不存在的节点类型、
# 连出环、把输入接到不存在的节点上——校验这些比校验一份扁平的计划书难得多，
# 而且出错时很难给出能喂回去的错误信息。
#
# 计划书 → 图的映射由**代码**完成（planner.plan_to_request），
# 模型碰不到图的结构。

PLAN_FIELD = {
    "type": "object",
    "required": ["name", "dtype"],
    "properties": {
        "name": {"type": "string"},
        "dtype": {"type": "string", "enum": list(VALID_DTYPES)},
        "description": {"type": "string"},
        "unit": {"type": "string"},
        "min": {"type": "number"},
        "max": {"type": "number"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "start": {"type": "string"},
        "end": {"type": "string"},
        "role": {"type": "string", "enum": list(VALID_ROLES)},
    },
}

PLAN_SPEC = {
    "type": "object",
    "required": ["name", "pipeline", "task_type", "fields", "num_records"],
    "properties": {
        "name": {"type": "string"},
        "pipeline": {"type": "string", "enum": ["tabular", "timeseries"]},
        "task_type": {
            "type": "string",
            "enum": ["classification", "regression", "time_series_forecast",
                     "clustering", "anomaly_detection"],
        },
        "domain": {"type": "string"},
        "num_records": {"type": "integer", "minimum": 1, "maximum": 2000000},
        "fields": {"type": "array", "items": PLAN_FIELD, "minItems": 1, "maxItems": 60},
        "target": {"type": "string"},
        "time_column": {"type": "string"},
        "missing_rate": {"type": "number", "minimum": 0, "maximum": 0.95},
        "noise_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        # 已知的合理强相关对，避免物理规律被误报成目标泄漏
        "known_correlated": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"},
                      "minItems": 2, "maxItems": 2},
        },
        "rationale": {"type": "string"},
        "open_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "why": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM_PLAN = """你在为一个合成数据生成器规划数据集结构。根据用户的自然语言需求，
产出一份**计划书**（不是代码、不是流程图）。

硬性要求：
1. 字段名用蛇形小写英文标识符（如 road_id、avg_speed），不要中文、不要空格。
2. 数值字段尽量给出 min / max，分类字段必须给出 categories，
   时间字段给出 start / end（ISO 日期）。
3. 若任务是时序预测，pipeline 填 timeseries，并且**必须**有一个 datetime 字段，
   把它的名字填进 time_column。
4. target 必须是 fields 里已有的字段名；没有明确预测目标就不要填。
5. 物理或业务上天然强相关的字段对（例如交通流里的 volume 与 occupancy）
   填进 known_correlated，避免被误判成目标泄漏。
6. 用户没说清楚的关键点（样本量、时间跨度、类别取值），
   **不要替他做主**：按合理默认值填，同时写进 open_questions。
7. rationale 用一两句中文说明你这样设计的依据。
8. 只输出 JSON，不要任何解释文字。"""
