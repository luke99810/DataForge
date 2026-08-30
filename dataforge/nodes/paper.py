"""论文解析与变量抽取。

链路（设计文档 §6）::

    PDF ─► 版面解析 ─► 章节定位 ─► LLM 抽取 ─► 变量卡 ─► ★人工确认 ─► Schema
           (规则)      (规则)      (LLM)                    (必经)      (代码)

**人工确认闸门不可跳过。** 变量抽错了后面全错，而且看不出来——
数据能生成、格式正确、质量分很高，只是它描述的不是那篇论文里的现象。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..graph.executor import NeedsHuman
from ..graph.node import NodeContext
from ..llm.contracts import (
    EXTRACT_VARIABLES, SYSTEM_EXTRACT, check_variable_semantics,
)

V = {
    "parse_pdf": "1.0.0",
    "extract_variables": "1.0.0",
    "confirm_variables": "1.0.0",
    "schema_from_cards": "1.0.0",
}

# 数据相关章节的标题特征。命中的段落优先送给模型，
# 整篇论文塞进去既贵又容易让模型抓到「相关工作」里的变量。
SECTION_PAT = re.compile(
    r"^\s*\d*\.?\d*\s*("
    r"dataset|datasets|data|data\s+description|experimental\s+set-?up|"
    r"experiments?|experimental\s+results|implementation\s+details|"
    r"数据集|数据|实验设置|实验|实现细节"
    r")\b",
    re.I,
)
RELATED_PAT = re.compile(r"^\s*\d*\.?\d*\s*(related\s+work|background|相关工作|研究现状)\b", re.I)


# ── SOURCE ────────────────────────────────────────────────

def parse_pdf(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """解析 PDF：逐页文本 + 章节切分 + 表格文本。

    先用规则定位数据相关章节，把送给模型的内容缩到几千 token。
    整篇塞进去不只是贵——「相关工作」里别人的变量会显著提高误抽率。
    """
    path = Path(params["path"])
    if not path.exists():
        raise FileNotFoundError(f"论文文件不存在：{path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要 PyMuPDF：pip install pymupdf") from e

    doc = fitz.open(path)
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append({"page": i, "text": text})
    doc.close()

    sections = _split_sections(pages)
    relevant = [s for s in sections if s["relevant"]]
    ctx.note(
        f"解析 {len(pages)} 页，切出 {len(sections)} 个章节，"
        f"其中 {len(relevant)} 个与数据相关",
        pages=len(pages), sections=len(sections), relevant=len(relevant),
    )
    return {
        "path": str(path),
        "title": _guess_title(pages),
        "pages": pages,
        "sections": sections,
        "char_count": sum(len(p["text"]) for p in pages),
    }


def _guess_title(pages: list[dict[str, Any]]) -> str:
    if not pages:
        return ""
    for line in pages[0]["text"].splitlines():
        t = line.strip()
        if len(t) > 12 and not t.lower().startswith(("arxiv", "doi", "http")):
            return t[:200]
    return ""


def _split_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按疑似小标题切章节，并标注是否与数据相关、是否属于相关工作。"""
    sections: list[dict[str, Any]] = []
    cur = {"title": "(前言)", "page": 1, "text": "", "relevant": False, "related_work": False}
    for p in pages:
        for line in p["text"].splitlines():
            stripped = line.strip()
            is_head = (
                0 < len(stripped) <= 90
                and (SECTION_PAT.match(stripped) or RELATED_PAT.match(stripped)
                     or re.match(r"^\s*\d+(\.\d+)*\s+\S", stripped))
            )
            if is_head:
                if cur["text"].strip():
                    sections.append(cur)
                cur = {
                    "title": stripped,
                    "page": p["page"],
                    "text": "",
                    "relevant": bool(SECTION_PAT.match(stripped)),
                    "related_work": bool(RELATED_PAT.match(stripped)),
                }
            else:
                cur["text"] += line + "\n"
    if cur["text"].strip():
        sections.append(cur)
    return sections


# ── EXTRACT ───────────────────────────────────────────────

MAX_CHARS = 14_000


def extract_variables(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """让模型从论文里抽变量卡。

    这是全流程唯一调用模型的节点。它的产物会进缓存——
    同一篇论文不必反复付费重抽；要重抽用 ``force``。
    """
    if ctx.llm is None:
        raise RuntimeError(
            "没有配置模型。请设置环境变量 DATAFORGE_API_KEY 后重试。\n"
            "（不会退化成规则抽取——那样抽出来的东西不该被当成论文里的变量。）"
        )

    paper = inputs["paper"]
    excerpt, cited = _build_excerpt(paper, int(params.get("max_chars", MAX_CHARS)))
    ctx.note(f"送入模型 {len(excerpt)} 字符，覆盖 {len(cited)} 个章节", sections=cited)

    attempts: list[dict[str, Any]] = []

    def on_attempt(n: int, err: str) -> None:
        attempts.append({"attempt": n, "error": err})
        if err:
            ctx.note(f"第 {n} 次输出不合契约，已把错误喂回重试", error=err)

    result = ctx.llm.structured(
        system=SYSTEM_EXTRACT,
        user=(
            f"论文标题：{paper.get('title', '(未知)')}\n\n"
            f"以下是论文中与数据相关的章节原文：\n\n{excerpt}\n\n"
            "请抽取本文实验使用的数据变量。"
        ),
        schema=EXTRACT_VARIABLES,
        max_retries=int(params.get("max_retries", 3)),
        on_attempt=on_attempt,
    )

    cards = result.get("variables", [])
    for c in cards:
        problems = check_variable_semantics(c)
        c["semantic_issues"] = problems
        # 出现在相关工作、或置信度过低的，默认不采纳——但保留让人看见
        c["suggest_adopt"] = (
            not problems
            and not c.get("from_related_work", False)
            and float(c.get("confidence", 0)) >= float(params.get("min_confidence", 0.5))
        )

    ctx.note(
        f"抽出 {len(cards)} 个候选变量，建议采纳 {sum(1 for c in cards if c['suggest_adopt'])} 个",
        total=len(cards),
    )
    return {
        "variables": cards,
        "task_type": result.get("task_type", "unknown"),
        "dataset_hint": result.get("dataset_hint", {}),
        "open_questions": result.get("open_questions", []),
        "llm_attempts": attempts,
        "llm_usage": ctx.llm.usage.to_dict(),
    }


def _build_excerpt(paper: dict[str, Any], budget: int) -> tuple[str, list[str]]:
    """挑相关章节拼成送模型的正文，控制在字符预算内。"""
    chunks: list[str] = []
    cited: list[str] = []
    used = 0

    # 相关章节优先；相关工作章节永远不送——那是误抽的主要来源
    ordered = sorted(
        (s for s in paper.get("sections", []) if not s.get("related_work")),
        key=lambda s: (not s.get("relevant"), s.get("page", 0)),
    )
    for s in ordered:
        body = s["text"].strip()
        if not body:
            continue
        piece = f"[第 {s['page']} 页 · {s['title']}]\n{body}\n"
        if used + len(piece) > budget:
            piece = piece[: max(0, budget - used)]
            if len(piece) < 200:
                break
        chunks.append(piece)
        cited.append(s["title"])
        used += len(piece)
        if used >= budget:
            break

    if not chunks:  # 章节切不出来时退回前几页原文
        text = "\n".join(p["text"] for p in paper.get("pages", [])[:4])
        return text[:budget], ["(全文前几页)"]
    return "\n".join(chunks), cited


# ── 人工确认闸门 ──────────────────────────────────────────

def confirm_variables(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """人工确认闸门。

    ``params["decisions"]`` 为空时抛 :class:`NeedsHuman` 停下；
    界面回填后，答案进入本节点的参数 → 指纹变化 → 本节点重算，
    而上游（解析、抽取）全部命中缓存。

    **续跑是缓存机制的自然结果，没有单独的「恢复」逻辑。**
    """
    ex = inputs["extraction"]
    cards = ex["variables"]
    decisions = params.get("decisions") or {}

    adopted_names = decisions.get("adopt")
    if adopted_names is None:
        # 还没人确认过 —— 停下来问
        raise NeedsHuman(
            "请确认要采纳哪些变量。抽错了后面全错，而且看不出来。",
            options=[c["name"] for c in cards],
            context={
                "variables": cards,
                "task_type": ex.get("task_type"),
                "open_questions": ex.get("open_questions", []),
                "suggested": [c["name"] for c in cards if c.get("suggest_adopt")],
            },
        )

    chosen = [c for c in cards if c["name"] in set(adopted_names)]
    if not chosen:
        raise ValueError("一个变量都没采纳，无法构建 Schema")

    # 覆盖项：人对单位、取值范围等的修正
    overrides = decisions.get("overrides") or {}
    for c in chosen:
        for k, v in (overrides.get(c["name"]) or {}).items():
            c[k] = v

    ctx.decision(
        "采纳哪些变量",
        {"adopted": [c["name"] for c in chosen],
         "rejected": [c["name"] for c in cards if c["name"] not in set(adopted_names)]},
        by="human",
        note="人工确认闸门",
    )
    for q, a in (decisions.get("answers") or {}).items():
        ctx.decision(q, a, by="human", note="论文未写明，由人拍板")

    for c in chosen:
        ctx.column_source(
            c["name"],
            origin="paper",
            unit=c.get("unit"),
            confidence=c.get("confidence"),
            source_span=c.get("source_span"),
        )

    return {
        "variables": chosen,
        "task_type": decisions.get("task_type") or ex.get("task_type", "unknown"),
        "answers": decisions.get("answers") or {},
        "rejected": [c["name"] for c in cards if c["name"] not in set(adopted_names)],
    }


# ── SCHEMA ────────────────────────────────────────────────

_DTYPE_MAP = {
    "integer": "integer", "float": "float", "string": "string",
    "boolean": "boolean", "datetime": "datetime", "categorical": "categorical",
}


def schema_from_cards(inputs: dict[str, Any], params: dict[str, Any], ctx: NodeContext) -> dict:
    """把确认过的变量卡转成内核认识的 Schema。"""
    from dataset_generator.core.schema import DatasetSchema, FieldSchema
    from dataset_generator.core.types import FieldType

    confirmed = inputs["confirmed"]
    fields = []
    target = None
    for c in confirmed["variables"]:
        fs = FieldSchema(
            name=c["name"],
            field_type=FieldType(_DTYPE_MAP[c["dtype"]]),
            description=c.get("description", "") or (c.get("source_span") or {}).get("quote", "")[:120],
        )
        if c.get("min") is not None:
            fs.constraints.min = c["min"]
        if c.get("max") is not None:
            fs.constraints.max = c["max"]
        if c.get("categories"):
            fs.constraints.categories = c["categories"]
        fields.append(fs)
        if c.get("role") == "target" and target is None:
            target = c["name"]

    schema = DatasetSchema(
        name=params.get("name", "paper_dataset"),
        description=params.get("description", ""),
        domain=params.get("domain", "custom"),
        fields=fields,
        target_field=params.get("target_field") or target,
    )
    errs = schema.validate()
    if errs:
        raise ValueError(f"Schema 非法：{errs}")
    ctx.note(f"由 {len(fields)} 个确认变量构建 Schema，目标字段 {schema.target_field}")
    return schema.to_dict()
