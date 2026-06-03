"""Schema definition and validation for datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import FieldType


@dataclass
class FieldConstraint:
    """Constraints for a single data field."""

    # Numeric constraints
    min: Optional[float] = None
    max: Optional[float] = None
    precision: int = 2

    # Categorical constraints
    categories: Optional[list[str]] = None
    weights: Optional[list[float]] = None  # probability weights for categories

    # String/text constraints
    pattern: Optional[str] = None  # regex or format string
    min_length: int = 0
    max_length: int = 256
    charset: str = "zh"  # 'zh', 'en', 'mixed'

    # Datetime constraints
    start: Optional[str] = None  # ISO 8601
    end: Optional[str] = None

    # Relational (fields that depend on others)
    depends_on: Optional[str] = None
    transform: Optional[str] = None  # python expression


@dataclass
class FieldSchema:
    """Definition of a single field in the dataset.

    Attributes:
        name: Field name (snake_case).
        field_type: Data type of this field.
        description: Human-readable description.
        nullable: Whether the field can contain null/missing values.
        unique: Whether values must be unique across records.
        constraints: Type-specific constraints.
        example: An example value for documentation.
    """

    name: str
    field_type: FieldType = FieldType.STRING
    description: str = ""
    nullable: bool = False
    unique: bool = False
    constraints: FieldConstraint = field(default_factory=FieldConstraint)
    example: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        result = {
            "name": self.name,
            "type": self.field_type.value,
            "description": self.description,
            "nullable": self.nullable,
            "unique": self.unique,
        }
        c = self.constraints
        if self.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            result["constraints"] = {
                "min": c.min,
                "max": c.max,
                "precision": c.precision,
            }
        elif self.field_type == FieldType.CATEGORICAL:
            result["constraints"] = {
                "categories": c.categories,
                "weights": c.weights,
            }
        elif self.field_type == FieldType.DATETIME:
            result["constraints"] = {
                "start": c.start,
                "end": c.end,
            }
        if self.example is not None:
            result["example"] = self.example
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldSchema":
        """Deserialize from a dictionary."""
        constraints = FieldConstraint()
        c = data.get("constraints", {})
        if c:
            constraints.min = c.get("min")
            constraints.max = c.get("max")
            constraints.precision = c.get("precision", 2)
            constraints.categories = c.get("categories")
            constraints.weights = c.get("weights")
            constraints.start = c.get("start")
            constraints.end = c.get("end")

        return cls(
            name=data["name"],
            field_type=FieldType(data.get("type", "string")),
            description=data.get("description", ""),
            nullable=data.get("nullable", False),
            unique=data.get("unique", False),
            constraints=constraints,
            example=data.get("example"),
        )


@dataclass
class DatasetSchema:
    """Complete schema definition for a dataset.

    Attributes:
        name: Dataset name.
        description: Full description of the dataset.
        domain: Knowledge domain.
        fields: List of field definitions.
        target_field: Name of the target/label field (if supervised).
        metadata: Arbitrary key-value metadata.
    """

    name: str
    description: str = ""
    domain: str = ""
    fields: list[FieldSchema] = field(default_factory=list)
    target_field: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def feature_fields(self) -> list[FieldSchema]:
        if self.target_field:
            return [f for f in self.fields if f.name != self.target_field]
        return list(self.fields)

    def validate(self) -> list[str]:
        """Validate schema consistency. Returns a list of issues (empty = valid)."""
        issues: list[str] = []
        names = set()
        for f in self.fields:
            if f.name in names:
                issues.append(f"Duplicate field name: {f.name}")
            names.add(f.name)
            if f.field_type == FieldType.CATEGORICAL and not f.constraints.categories:
                issues.append(f"Categorical field '{f.name}' has no categories defined")
        if self.target_field and self.target_field not in self.field_names:
            issues.append(f"target_field '{self.target_field}' not found in fields")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "target_field": self.target_field,
            "fields": [f.to_dict() for f in self.fields],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSchema":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain", ""),
            fields=[FieldSchema.from_dict(f) for f in data.get("fields", [])],
            target_field=data.get("target_field"),
            metadata=data.get("metadata", {}),
        )
