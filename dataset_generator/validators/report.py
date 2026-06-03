"""Validation report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from .quality import DataQualityValidator, QualityIssue


@dataclass
class ValidationReport:
    """Structured validation report."""

    dataset_name: str
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_checks: int = 0
    errors: list[QualityIssue] = field(default_factory=list)
    warnings: list[QualityIssue] = field(default_factory=list)
    info: list[QualityIssue] = field(default_factory=list)
    overall_status: str = "PASS"

    @classmethod
    def from_validator(cls, name: str, validator: DataQualityValidator) -> "ValidationReport":
        """Build a report from a completed validator instance."""
        report = cls(dataset_name=name)
        report.total_checks = len(validator.issues)

        for issue in validator.issues:
            if issue.severity == "error":
                report.errors.append(issue)
            elif issue.severity == "warning":
                report.warnings.append(issue)
            else:
                report.info.append(issue)

        report.overall_status = "PASS" if len(report.errors) == 0 else "FAIL"
        return report

    def to_markdown(self) -> str:
        """Render as a Markdown report string."""
        lines = [
            f"# Data Quality Report: {self.dataset_name}",
            f"",
            f"**Validated at:** {self.validated_at}",
            f"**Overall status:** {'✅ PASS' if self.overall_status == 'PASS' else '❌ FAIL'}",
            f"**Total checks:** {self.total_checks}",
            f"**Errors:** {len(self.errors)} | **Warnings:** {len(self.warnings)} | **Info:** {len(self.info)}",
            f"",
        ]

        if self.errors:
            lines.append("## ❌ Errors")
            lines.append("")
            for e in self.errors:
                col = f" (`{e.column}`)" if e.column else ""
                lines.append(f"- **[{e.category}]**{col}: {e.message}")
            lines.append("")

        if self.warnings:
            lines.append("## ⚠️ Warnings")
            lines.append("")
            for w in self.warnings:
                col = f" (`{w.column}`)" if w.column else ""
                lines.append(f"- **[{w.category}]**{col}: {w.message}")
            lines.append("")

        if self.info:
            lines.append("## ℹ️ Info")
            lines.append("")
            for i_info in self.info:
                lines.append(f"- **[{i_info.category}]**: {i_info.message}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize as JSON string."""
        return json.dumps({
            "dataset_name": self.dataset_name,
            "validated_at": self.validated_at,
            "total_checks": self.total_checks,
            "overall_status": self.overall_status,
            "errors": [{"category": e.category, "message": e.message, "column": e.column} for e in self.errors],
            "warnings": [{"category": w.category, "message": w.message, "column": w.column} for w in self.warnings],
        }, ensure_ascii=False, indent=2)
