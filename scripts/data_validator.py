#!/usr/bin/env python3
"""
Data Validator — 数据质量验证工具

对生成的数据集进行完整性、一致性、分布和格式检查。
输出详细的验证报告。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_data(file_path: str) -> list[dict]:
    """加载数据文件。"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        import csv
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    elif suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]

    elif suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_schema(schema_path: str) -> dict:
    """加载 Schema。"""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ValidationReport:
    """验证报告生成器。"""

    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)

    def summary(self) -> dict:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        return {
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "warnings": len(self.warnings),
            "errors": len(self.errors),
        }

    def to_markdown(self) -> str:
        lines = []
        s = self.summary()
        lines.append("# Data Validation Report\n")
        lines.append(f"**Summary**: {s['passed']}/{s['total_checks']} checks passed")
        if s["errors"]:
            lines.append(f", {s['errors']} errors")
        if s["warnings"]:
            lines.append(f", {s['warnings']} warnings")
        lines.append("\n\n")

        if self.errors:
            lines.append("## Errors\n")
            for e in self.errors:
                lines.append(f"- ❌ {e}")
            lines.append("\n")

        if self.warnings:
            lines.append("## Warnings\n")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("\n")

        lines.append("## Checks\n")
        for c in self.checks:
            icon = "✅" if c["passed"] else "❌"
            lines.append(f"- {icon} **{c['name']}**")
            if c["detail"]:
                lines.append(f"  {c['detail']}")
        lines.append("")

        return "\n".join(lines)


def validate_completeness(data: list, schema: dict, report: ValidationReport) -> None:
    """完整性检查：必填字段、数据量、缺失值。"""
    fields = schema.get("fields", [])
    required = [f["name"] for f in fields if f.get("required", False)]

    # 数据量检查
    expected = schema.get("expected_count")
    report.add_check(
        "Record Count",
        expected is None or len(data) >= expected,
        f"Expected >= {expected}, got {len(data)}" if expected else f"Total records: {len(data)}",
    )

    if not data:
        report.add_error("Dataset is empty")
        return

    # 必填字段检查
    for field_name in required:
        missing_count = sum(1 for r in data if r.get(field_name) is None)
        if missing_count > 0:
            report.add_error(
                f"Required field '{field_name}' has {missing_count} missing values "
                f"({missing_count / len(data) * 100:.1f}%)"
            )
        else:
            report.add_check(f"Required field '{field_name}'", True, "No missing values")

    # 字段覆盖检查
    all_keys = set()
    for r in data:
        all_keys.update(r.keys())
    schema_keys = {f["name"] for f in fields}
    extra_keys = all_keys - schema_keys
    if extra_keys:
        report.add_warning(f"Extra fields not in schema: {', '.join(extra_keys)}")


def validate_consistency(data: list, schema: dict, report: ValidationReport) -> None:
    """一致性检查：类型、取值范围、关系。"""
    fields = schema.get("fields", [])

    for field in fields:
        name = field["name"]
        ftype = field.get("type", "string")
        constraints = field.get("constraints", {})

        values = [r.get(name) for r in data if r.get(name) is not None]

        if not values:
            report.add_warning(f"Field '{name}' has no non-null values")
            continue

        # 类型检查
        type_ok = True
        if ftype == "integer" or ftype == "float":
            type_ok = all(isinstance(v, (int, float)) for v in values)
        elif ftype == "string":
            type_ok = all(isinstance(v, str) for v in values)
        elif ftype == "boolean":
            type_ok = all(isinstance(v, bool) for v in values)

        report.add_check(
            f"Type check: {name} ({ftype})",
            type_ok,
            f"All {len(values)} values" if type_ok else "Type mismatch found",
        )

        # 范围检查
        if ftype in ("integer", "float"):
            lo = constraints.get("min")
            hi = constraints.get("max")
            if lo is not None and hi is not None:
                out_of_range = sum(1 for v in values if v < lo or v > hi)
                report.add_check(
                    f"Range check: {name} [{lo}, {hi}]",
                    out_of_range == 0,
                    f"{out_of_range} out of range" if out_of_range > 0 else "All in range",
                )

        # 分类值检查
        if ftype == "categorical":
            expected_cats = set(constraints.get("categories", []))
            actual_cats = set(values)
            extra = actual_cats - expected_cats
            report.add_check(
                f"Category check: {name}",
                len(extra) == 0,
                f"Unexpected categories: {extra}" if extra else f"{len(expected_cats)} categories validated",
            )


def validate_distribution(data: list, schema: dict, report: ValidationReport) -> None:
    """分布检查：数值分布、类别均衡性。"""
    fields = schema.get("fields", [])

    for field in fields:
        name = field["name"]
        ftype = field.get("type", "string")
        values = [r.get(name) for r in data if r.get(name) is not None]

        if not values:
            continue

        if ftype in ("integer", "float"):
            import statistics
            try:
                mean_val = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                report.add_check(
                    f"Distribution: {name}",
                    True,
                    f"mean={mean_val:.2f}, std={stdev:.2f}, n={len(values)}",
                )
            except Exception:
                pass

        elif ftype == "categorical":
            counter = Counter(values)
            total = len(values)
            distribution = {k: f"{v}/{total} ({v/total*100:.1f}%)" for k, v in counter.most_common()}
            report.add_check(
                f"Category distribution: {name}",
                True,
                "; ".join(f"{k}: {v}" for k, v in distribution.items()),
            )


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--schema", required=True, help="Path to schema file")
    parser.add_argument("--report", default="validation_report.md", help="Output report path")
    args = parser.parse_args()

    report = ValidationReport()

    print(f"Loading data from {args.data}...")
    try:
        data = load_data(args.data)
        report.add_check("File loading", True, f"Loaded {len(data)} records")
    except Exception as e:
        report.add_error(f"Failed to load data: {e}")
        print(report.to_markdown())
        sys.exit(1)

    print(f"Loading schema from {args.schema}...")
    try:
        schema = load_schema(args.schema)
        report.add_check("Schema loading", True, f"{len(schema.get('fields', []))} fields defined")
    except Exception as e:
        report.add_error(f"Failed to load schema: {e}")
        print(report.to_markdown())
        sys.exit(1)

    print("Running validation checks...")
    validate_completeness(data, schema, report)
    validate_consistency(data, schema, report)
    validate_distribution(data, schema, report)

    # 输出报告
    markdown_report = report.to_markdown()
    print("\n" + markdown_report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"Report saved to {args.report}")

    # 退出码
    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
