"""File I/O utilities for dataset reading and writing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..core.types import OutputFormat

logger = logging.getLogger(__name__)


def write_output(
    df: pd.DataFrame,
    path: str | Path,
    fmt: OutputFormat | str = OutputFormat.CSV,
) -> Path:
    """Write a DataFrame to the specified format.

    Args:
        df: DataFrame to write.
        path: Output file path.
        fmt: Output format.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    if isinstance(fmt, str):
        fmt = OutputFormat(fmt)

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == OutputFormat.CSV:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif fmt == OutputFormat.JSON:
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    elif fmt == OutputFormat.JSONL:
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    elif fmt == OutputFormat.PARQUET:
        df.to_parquet(path, index=False)
    elif fmt == OutputFormat.EXCEL:
        df.to_excel(path, index=False, engine="openpyxl")
    elif fmt == OutputFormat.MARKDOWN:
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.head(1000).to_markdown(index=False))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    size_kb = path.stat().st_size / 1024
    logger.info("Wrote %d records to %s (%.1f KB)", len(df), path, size_kb)
    return path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a file, auto-detecting format."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    elif ext == ".json":
        return pd.read_json(path, orient="records")
    elif ext == ".jsonl":
        return pd.read_json(path, lines=True)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Cannot auto-detect format for: {ext}")
