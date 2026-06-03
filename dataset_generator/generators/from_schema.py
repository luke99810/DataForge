"""Schema-driven data generator (reads JSON/YAML schema files)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ..core.config import DatasetConfig
from ..core.schema import DatasetSchema, FieldSchema
from ..core.types import FieldType
from .tabular import TabularGenerator
from .timeseries import TimeseriesGenerator

logger = logging.getLogger(__name__)


class SchemaDrivenGenerator:
    """Generate datasets from external schema definitions (JSON/YAML).

    This generator bridges the gap between document extraction and data
    generation. It reads a schema file, builds the internal schema objects,
    and delegates to the appropriate generator.
    """

    def __init__(self, schema_path: str | Path, config: DatasetConfig) -> None:
        self.schema_path = Path(schema_path)
        self.config = config
        self.schema = self._load_schema()

    def _load_schema(self) -> DatasetSchema:
        """Load and parse the schema file."""
        ext = self.schema_path.suffix.lower()

        if ext == ".json":
            with open(self.schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif ext in (".yaml", ".yml"):
            with open(self.schema_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported schema format: {ext}")

        if not isinstance(data, dict):
            raise ValueError("Schema must be a JSON/YAML object")

        # Handle both raw field lists and full schema objects
        if "fields" in data:
            schema = DatasetSchema.from_dict(data)
        else:
            # Treat as a flat list of fields
            fields = []
            if isinstance(data, list):
                field_dicts = data
            else:
                field_dicts = [{"name": k, **v} for k, v in data.items()]

            for fd in field_dicts:
                if isinstance(fd, str):
                    fields.append(FieldSchema(name=fd, field_type=FieldType.STRING))
                else:
                    fields.append(FieldSchema.from_dict(fd))

            schema = DatasetSchema(
                name=self.schema_path.stem,
                fields=fields,
                description=f"Generated from {self.schema_path.name}",
            )

        # Validate
        issues = schema.validate()
        if issues:
            logger.warning("Schema issues: %s", issues)

        return schema

    def generate(self) -> "pd.DataFrame":
        """Generate data using the appropriate generator."""
        from ..core.types import TaskType

        # Auto-detect if this is time-series data
        has_timestamp = any(
            f.name.lower() in ("timestamp", "time", "date", "datetime", "时间")
            or f.field_type == FieldType.DATETIME
            for f in self.schema.fields
        )

        if has_timestamp or self.config.task_type == TaskType.TIME_SERIES_FORECAST:
            gen = TimeseriesGenerator(self.config, self.schema)
        else:
            gen = TabularGenerator(self.config, self.schema)

        return gen.generate()

    @classmethod
    def from_extracted_fields(
        cls, fields: list[dict[str, Any]], config: DatasetConfig
    ) -> "SchemaDrivenGenerator":
        """Create a generator from extracted field patterns (e.g., from document parsing).

        Typically called after document_parser.extract_fields().
        """
        field_schemas = []
        for f in fields:
            field_schemas.append(FieldSchema.from_dict(f))

        schema = DatasetSchema(
            name=config.project_name,
            fields=field_schemas,
            description=config.description,
        )

        # Create a temp schema file
        tmp_path = Path(config.output_dir) / "_temp_schema.json"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(schema.to_dict(), f, ensure_ascii=False, indent=2)

        return cls(tmp_path, config)
