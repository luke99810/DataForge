"""Tabular data generator for structured datasets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..core.schema import FieldSchema
from ..core.types import FieldType
from .base import BaseGenerator

logger = logging.getLogger(__name__)


class TabularGenerator(BaseGenerator):
    """Generate tabular datasets (CSV, JSON, Parquet, Excel).

    Supports all field types: integer, float, string, boolean, datetime,
    categorical, text. Handles field dependencies and distribution constraints.
    """

    def _generate_records(self) -> list[dict[str, Any]]:
        n = self.config.num_records
        records = [{} for _ in range(n)]

        for field in self.schema.fields:
            values = self._generate_column(field, n)
            for i, val in enumerate(values):
                records[i][field.name] = val

        # Apply cross-field dependencies
        self._apply_dependencies(records)

        return records

    def _generate_column(self, field: FieldSchema, n: int) -> list[Any]:
        """Generate values for a single column."""
        c = field.constraints
        ft = field.field_type

        if ft == FieldType.INTEGER:
            lo = int(c.min) if c.min is not None else 0
            hi = int(c.max) if c.max is not None else 100
            values = self.rng.integers(lo, hi + 1, size=n).tolist()

        elif ft == FieldType.FLOAT:
            lo = c.min if c.min is not None else 0.0
            hi = c.max if c.max is not None else 1.0
            values = self.rng.uniform(lo, hi, size=n)
            values = [round(v, c.precision) for v in values]

        elif ft == FieldType.BOOLEAN:
            values = self.rng.choice([True, False], size=n).tolist()

        elif ft == FieldType.CATEGORICAL:
            cats = c.categories or ["A", "B", "C"]
            weights = c.weights if c.weights else None
            values = self.rng.choice(cats, size=n, p=weights).tolist()

        elif ft == FieldType.DATETIME:
            start = pd.Timestamp(c.start or "2024-01-01")
            end = pd.Timestamp(c.end or "2024-12-31")
            delta = (end - start).days
            offsets = self.rng.integers(0, delta + 1, size=n)
            values = [(start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in offsets]

        elif ft == FieldType.STRING:
            if c.pattern:
                values = [c.pattern.format(i=i) for i in range(1, n + 1)]
            else:
                values = [f"{field.name}_{i:06d}" for i in range(1, n + 1)]

        elif ft == FieldType.TEXT:
            templates = [
                "这是关于{field.name}的第{i}条记录描述",
                "This is record {i} for field {field.name}",
            ]
            tmpl = templates[0] if self.config.language == "zh" else templates[1]
            values = [tmpl.format(i=i) for i in range(1, n + 1)]

        else:
            values = [None] * n

        # Apply uniqueness if required
        if field.unique:
            seen: set[Any] = set()
            unique_vals: list[Any] = []
            base = 0
            for v in values:
                while v in seen:
                    if isinstance(v, (int, float)):
                        v = v + 1
                    elif isinstance(v, str):
                        base += 1
                        v = f"{v}_{base}"
                    else:
                        base += 1
                        v = base
                seen.add(v)
                unique_vals.append(v)
            values = unique_vals

        return values

    def _apply_dependencies(self, records: list[dict[str, Any]]) -> None:
        """Apply inter-field dependency transformations."""
        for field in self.schema.fields:
            dep = field.constraints.depends_on
            transform = field.constraints.transform
            if dep and transform:
                for rec in records:
                    if dep in rec:
                        try:
                            local_vars = {"x": rec[dep], "np": np}
                            rec[field.name] = eval(transform, {"__builtins__": {}}, local_vars)
                        except Exception:
                            pass
