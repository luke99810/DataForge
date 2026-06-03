"""Extract data patterns from documents (PDF, Word, Markdown, plain text)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extract data structure patterns from document files.

    Supports reading files directly (TXT, MD) and provides pattern
    detection for table headers, field definitions, data samples,
    and structural relationships.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.text = ""
        self._load()

    def _load(self) -> None:
        """Load document content based on file type."""
        ext = self.file_path.suffix.lower()

        if ext in (".txt", ".md", ".csv", ".json", ".jsonl", ".yaml", ".yml"):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
        elif ext == ".pdf":
            raise NotImplementedError(
                "PDF reading requires the 'pdf' skill. Load it first via Skill tool, "
                "then pass extracted text to DocumentExtractor.from_text()."
            )
        elif ext in (".docx", ".doc"):
            raise NotImplementedError(
                "Word document reading requires the 'docx' skill. Load it first, "
                "then pass extracted text to DocumentExtractor.from_text()."
            )
        else:
            logger.warning("Unknown format '%s', trying as plain text", ext)
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                self.text = f.read()

    @classmethod
    def from_text(cls, text: str) -> "DocumentExtractor":
        """Create extractor from already-read text content."""
        instance = cls.__new__(cls)
        instance.file_path = Path("inline")
        instance.text = text
        return instance

    def extract_fields(self) -> list[dict[str, Any]]:
        """Extract field definitions from the document.

        Detects multiple input styles:
        - Markdown tables
        - Key-value definitions (name: type / description)
        - CSV/TSV header rows
        - Bullet-point field lists
        - Field: type, description format
        """
        fields: list[dict[str, Any]] = []

        # Strategy 1: Markdown table
        table_fields = self._detect_markdown_table()
        if len(table_fields) >= 2:
            return table_fields

        # Strategy 2: Key-value definitions
        kv_fields = self._detect_key_value_fields()
        if kv_fields:
            return kv_fields

        # Strategy 3: CSV header
        csv_fields = self._detect_csv_header()
        if csv_fields:
            return csv_fields

        logger.warning("No field patterns detected in document")
        return fields

    def extract_data_sample(self) -> list[dict[str, Any]] | None:
        """Extract sample data rows from the document.

        Returns a list of record dicts if data rows are found.
        """
        lines = self.text.strip().split("\n")
        data_rows: list[dict[str, Any]] = []

        # Try to find a CSV-like data section
        in_data = False
        headers: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Detect header
            if not in_data and "," in stripped:
                parts = self._split_csv_line(stripped)
                if len(parts) >= 3 and all(not p.replace(".", "").replace("-", "").isdigit() for p in parts[:3]):
                    headers = parts
                    in_data = True
                    continue

            # Data rows
            if in_data and "," in stripped:
                parts = self._split_csv_line(stripped)
                if len(parts) == len(headers):
                    row = {}
                    for h, v in zip(headers, parts):
                        row[h] = self._infer_value_type(v)
                    data_rows.append(row)

        return data_rows if data_rows else None

    def generate_schema_json(self, output_path: str | Path) -> dict[str, Any]:
        """Generate a JSON schema file from extracted patterns."""
        fields = self.extract_fields()
        data_sample = self.extract_data_sample()

        schema = {
            "name": self.file_path.stem,
            "description": f"Schema extracted from {self.file_path.name}",
            "fields": fields,
            "sample_count": len(data_sample) if data_sample else 0,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

        return schema

    # --- Private detection methods ---

    def _detect_markdown_table(self) -> list[dict[str, Any]]:
        """Detect and parse Markdown tables."""
        lines = self.text.strip().split("\n")
        fields: list[dict[str, Any]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "|" not in stripped:
                continue

            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if not cells:
                continue

            # Skip separator lines like |---|---|
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue

            # This is a header or data row
            for cell in cells:
                field = self._classify_column_header(cell)
                fields.append(field)
            break  # Only process the first table

        return fields

    def _detect_key_value_fields(self) -> list[dict[str, Any]]:
        """Detect field definitions in key:type or key:description format."""
        fields: list[dict[str, Any]] = []
        patterns = [
            r"^[-*]\s*`?(\w+)`?\s*[:：]\s*(.+)$",
            r"^(\w+)\s*[:：]\s*(.+)$",
        ]

        for line in self.text.strip().split("\n"):
            for pat in patterns:
                match = re.match(pat, line.strip())
                if match:
                    name = match.group(1)
                    rest = match.group(2).strip()
                    fields.append(self._classify_column_header(name, rest))
                    break
            if len(fields) >= 3:
                break

        return fields

    def _detect_csv_header(self) -> list[dict[str, Any]]:
        """Detect CSV header row."""
        for line in self.text.strip().split("\n")[:5]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "," in stripped:
                parts = self._split_csv_line(stripped)
                if len(parts) >= 2:
                    return [self._classify_column_header(p) for p in parts]
        return []

    def _classify_column_header(self, name: str, description: str = "") -> dict[str, Any]:
        """Infer field type from column name."""
        name_lower = name.lower()
        field: dict[str, Any] = {"name": name, "description": description}

        if any(kw in name_lower for kw in ["id", "编号", "序号", "code"]):
            field["type"] = "string"
            field["constraints"] = {"pattern": f"{name}_%06d"}
        elif any(kw in name_lower for kw in ["age", "年龄", "count", "数量", "num", "号"]):
            field["type"] = "integer"
            field["constraints"] = {"min": 0, "max": 100}
        elif any(kw in name_lower for kw in ["price", "价格", "amount", "金额", "rate", "ratio", "比例", "score", "得分", "值", "value", "speed", "速度", "flow", "流量"]):
            field["type"] = "float"
            field["constraints"] = {"min": 0.0, "max": 1000.0, "precision": 2}
        elif any(kw in name_lower for kw in ["date", "日期", "time", "时间", "timestamp"]):
            field["type"] = "datetime"
            field["constraints"] = {"start": "2024-01-01", "end": "2024-12-31"}
        elif any(kw in name_lower for kw in ["is_", "has_", "是否", "flag", "标志", "enable"]):
            field["type"] = "boolean"
        elif any(kw in name_lower for kw in ["category", "类别", "type", "类型", "status", "状态", "label", "标签", "class"]):
            field["type"] = "categorical"
            field["constraints"] = {"categories": ["A", "B", "C", "D"]}
        elif any(kw in name_lower for kw in ["desc", "描述", "content", "内容", "text", "文本", "body"]):
            field["type"] = "text"
        else:
            field["type"] = "string"

        return field

    @staticmethod
    def _split_csv_line(line: str) -> list[str]:
        """Split a CSV line, handling quoted fields."""
        parts = []
        current = ""
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == "," and not in_quotes:
                parts.append(current.strip().strip('"'))
                current = ""
            else:
                current += ch
        parts.append(current.strip().strip('"'))
        return parts

    @staticmethod
    def _infer_value_type(value: str) -> str | int | float | bool:
        """Infer and convert a string value to its likely Python type."""
        v = value.strip()
        if v.lower() in ("true", "false", "yes", "no"):
            return v.lower() in ("true", "yes")
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v
