#!/usr/bin/env python3
"""
Document Parser — 文档数据模式提取器

从 PDF、Word、Markdown 文档中提取数据结构、字段定义和数据模式，
用于后续数据生成的 Schema 构建。
"""

import argparse
import json
import re
from pathlib import Path


def detect_table_headers(text: str) -> list[dict]:
    """检测文本中的表格/字段定义。"""
    fields = []
    lines = text.strip().split("\n")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 模式1: "字段名: 类型, 描述" 格式
        match = re.match(r"^[\*\-]\s*['\"]?(\w+)['\"]?\s*[:：]\s*(.+)$", line)
        if match:
            name = match.group(1)
            desc = match.group(2)
            fields.append({"name": name, "description": desc, "source_line": i + 1})
            continue

        # 模式2: Markdown 表格行
        if "|" in line and i > 0 and "---" not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells and not fields:
                for cell in cells:
                    fields.append({"name": cell, "description": "", "source_line": i + 1})
                break

    return fields


def detect_data_types(text: str, field_names: list[str]) -> dict:
    """推断字段的数据类型。"""
    type_hints = {}

    for name in field_names:
        name_lower = name.lower()

        if any(kw in name_lower for kw in ["id", "编号", "序号"]):
            type_hints[name] = {"type": "string", "constraints": {"pattern": f"{name}_{{:06d}}"}}
        elif any(kw in name_lower for kw in ["age", "年龄", "count", "数量", "num"]):
            type_hints[name] = {"type": "integer", "constraints": {"min": 0, "max": 100}}
        elif any(kw in name_lower for kw in ["price", "价格", "amount", "金额", "rate", "ratio", "比例", "score", "得分"]):
            type_hints[name] = {"type": "float", "constraints": {"min": 0.0, "max": 1000000.0, "precision": 2}}
        elif any(kw in name_lower for kw in ["date", "日期", "time", "时间", "timestamp"]):
            type_hints[name] = {"type": "datetime", "constraints": {"start": "2024-01-01", "end": "2024-12-31"}}
        elif any(kw in name_lower for kw in ["is_", "has_", "是否", "flag", "标志"]):
            type_hints[name] = {"type": "boolean"}
        elif any(kw in name_lower for kw in ["category", "类别", "type", "类型", "status", "状态", "label", "标签"]):
            type_hints[name] = {"type": "categorical", "constraints": {"categories": ["A", "B", "C"]}}
        elif any(kw in name_lower for kw in ["name", "姓名", "名称", "title", "标题", "desc", "描述"]):
            type_hints[name] = {"type": "string"}
        else:
            type_hints[name] = {"type": "string"}

    return type_hints


def detect_data_patterns(text: str) -> list[str]:
    """检测数据样本模式（如列表中的示例值）。"""
    patterns = []

    # 检测 CSV/TSV 样式的数据行
    lines = text.strip().split("\n")
    in_data_section = False
    for line in lines:
        stripped = line.strip()
        if "," in stripped and not stripped.startswith("#"):
            parts = [p.strip().strip('"') for p in stripped.split(",")]
            if len(parts) >= 2:
                patterns.append(parts)

    return patterns


def extract_from_text(text: str) -> dict:
    """从文本中提取数据 schema。"""
    fields_raw = detect_table_headers(text)
    field_names = [f["name"] for f in fields_raw]

    if not field_names:
        # 尝试从数据模式中推断
        patterns = detect_data_patterns(text)
        if patterns:
            field_names = [f"field_{i}" for i in range(len(patterns[0]))]
        else:
            return {"fields": [], "note": "Could not detect data structure"}

    type_hints = detect_data_types(text, field_names)

    fields = []
    for f in fields_raw:
        name = f["name"]
        field_def = {"name": name, "description": f.get("description", "")}
        if name in type_hints:
            field_def.update(type_hints[name])
        fields.append(field_def)

    # 补充未在 header 中出现的字段
    for name in field_names:
        if name not in [f["name"] for f in fields]:
            field_def = {"name": name}
            if name in type_hints:
                field_def.update(type_hints[name])
            fields.append(field_def)

    return {"fields": fields, "source_fields_count": len(fields)}


def main():
    parser = argparse.ArgumentParser(description="Extract data patterns from documents")
    parser.add_argument("--input", required=True, help="Input document path")
    parser.add_argument("--output", default="extracted_schema.json", help="Output schema path")
    parser.add_argument("--format", default="auto", choices=["auto", "txt", "md", "csv"], help="Input format hint")
    args = parser.parse_args()

    input_path = Path(args.input)
    print(f"Reading {input_path}...")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 如果是 Markdown，先做简单清洗
    if args.format == "md" or input_path.suffix == ".md" or args.format == "auto":
        # 移除代码块
        text = re.sub(r"```[\s\S]*?```", "", text)

    print(f"Extracting data patterns from {len(text)} characters...")
    schema = extract_from_text(text)

    # 保存 schema
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(schema['fields'])} fields")
    print(f"Schema saved to {args.output}")

    # 输出字段列表
    print("\nDetected Fields:")
    for field in schema["fields"]:
        print(f"  - {field['name']} ({field.get('type', 'string')}): {field.get('description', '')}")


if __name__ == "__main__":
    main()
