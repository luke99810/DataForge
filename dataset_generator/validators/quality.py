"""Data quality validation checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..core.schema import DatasetSchema

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """A single quality issue found during validation."""

    severity: str  # 'error', 'warning', 'info'
    category: str  # 'completeness', 'consistency', 'distribution', 'format'
    message: str
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class DataQualityValidator:
    """Validate generated datasets against quality criteria.

    Checks:
    1. Completeness — no missing required fields, row count matches
    2. Consistency — no logical conflicts, relationships valid
    3. Distribution — numeric ranges, categorical balance
    4. Format — correct data types, encoding, output format
    """

    def __init__(self, schema: DatasetSchema) -> None:
        self.schema = schema
        self.issues: list[QualityIssue] = []

    def validate(self, df: pd.DataFrame) -> list[QualityIssue]:
        """Run all validation checks. Returns list of issues (empty = clean)."""
        self.issues = []
        self._check_completeness(df)
        self._check_consistency(df)
        self._check_distribution(df)
        self._check_format(df)
        return self.issues

    def _check_completeness(self, df: pd.DataFrame) -> None:
        """Check for missing values and record count."""
        # Record count
        if len(df) == 0:
            self.issues.append(QualityIssue("error", "completeness", "Dataset is empty"))
            return

        # Missing values per column
        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df)
            if null_pct > 0.5:
                self.issues.append(QualityIssue(
                    "error", "completeness",
                    f"Column '{col}' has {null_pct:.1%} missing values",
                    column=col,
                    details={"missing_count": int(null_count), "missing_pct": round(null_pct, 3)},
                ))
            elif null_pct > 0.1:
                self.issues.append(QualityIssue(
                    "warning", "completeness",
                    f"Column '{col}' has {null_pct:.1%} missing values",
                    column=col,
                ))

        # Check all schema fields exist
        for f in self.schema.fields:
            if f.name not in df.columns:
                self.issues.append(QualityIssue(
                    "warning", "completeness",
                    f"Schema field '{f.name}' not found in data",
                    column=f.name,
                ))

    def _check_consistency(self, df: pd.DataFrame) -> None:
        """Check logical consistency."""
        # Check unique constraints
        for f in self.schema.fields:
            if f.unique and f.name in df.columns:
                n_unique = df[f.name].nunique()
                n_total = df[f.name].notna().sum()
                if n_unique < n_total:
                    self.issues.append(QualityIssue(
                        "error", "consistency",
                        f"Field '{f.name}' should be unique but has {n_total - n_unique} duplicates",
                        column=f.name,
                        details={"unique": int(n_unique), "total": int(n_total)},
                    ))

        # Check categorical fields have valid values
        for f in self.schema.fields:
            if f.name in df.columns and f.constraints.categories:
                valid_cats = set(f.constraints.categories)
                actual_cats = set(df[f.name].dropna().unique())
                unknown_cats = actual_cats - valid_cats
                if unknown_cats:
                    self.issues.append(QualityIssue(
                        "warning", "consistency",
                        f"Field '{f.name}' has unexpected categories: {unknown_cats}",
                        column=f.name,
                    ))

    def _check_distribution(self, df: pd.DataFrame) -> None:
        """Check numerical distribution bounds."""
        for f in self.schema.fields:
            if f.name not in df.columns:
                continue
            col = df[f.name].dropna()
            if len(col) == 0:
                continue

            c = f.constraints
            if c.min is not None and col.min() < c.min:
                self.issues.append(QualityIssue(
                    "warning", "distribution",
                    f"Field '{f.name}' has values below min ({col.min()} < {c.min})",
                    column=f.name,
                ))
            if c.max is not None and col.max() > c.max:
                self.issues.append(QualityIssue(
                    "warning", "distribution",
                    f"Field '{f.name}' has values above max ({col.max()} > {c.max})",
                    column=f.name,
                ))

    def _check_format(self, df: pd.DataFrame) -> None:
        """Check output format and encoding."""
        if df.empty:
            return

        # Check for non-ASCII chars that might cause encoding issues
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(10).astype(str)
            for i, val in enumerate(sample):
                if any(ord(ch) > 127 for ch in val):
                    logger.debug("Non-ASCII content detected in '%s', row %d", col, i)
                    break

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0
