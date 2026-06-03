"""Configuration management for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import Domain, TaskType, OutputFormat


@dataclass
class DatasetConfig:
    """Master configuration for a dataset generation task.

    Attributes:
        project_name: Unique project identifier.
        domain: Knowledge domain of the dataset.
        task_type: ML task the dataset targets.
        description: Human-readable description of the dataset purpose.
        num_records: Target number of records to generate.
        output_format: Primary output file format.
        output_dir: Directory to write output files.
        seed: Random seed for reproducibility.
        noise_ratio: Proportion of noise to inject (0.0 to 1.0).
        missing_rate: Proportion of missing values (0.0 to 1.0).
        language: Primary language for text data ('zh' or 'en').
        literature_sources: Number of papers to search for feature extraction.
        generate_github_project: Whether to scaffold a GitHub project structure.
        custom_constraints: Arbitrary key-value constraints for custom domains.
    """

    project_name: str = "dataset"
    domain: Domain = Domain.CUSTOM
    task_type: TaskType = TaskType.CLASSIFICATION
    description: str = ""
    num_records: int = 1000
    output_format: OutputFormat = OutputFormat.CSV
    output_dir: str = "./output"
    seed: int = 42
    noise_ratio: float = 0.0
    missing_rate: float = 0.0
    language: str = "zh"
    literature_sources: int = 5
    generate_github_project: bool = True
    custom_constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not (0.0 <= self.noise_ratio <= 1.0):
            raise ValueError(f"noise_ratio must be 0.0-1.0, got {self.noise_ratio}")
        if not (0.0 <= self.missing_rate <= 1.0):
            raise ValueError(f"missing_rate must be 0.0-1.0, got {self.missing_rate}")
        if self.num_records < 1:
            raise ValueError(f"num_records must be >= 1, got {self.num_records}")
        if self.language not in ("zh", "en"):
            raise ValueError(f"language must be 'zh' or 'en', got {self.language}")

    @classmethod
    def estimate_size(cls, domain: Domain, task: TaskType, complexity: str = "prototype") -> int:
        """Auto-estimate dataset size based on domain, task, and complexity.

        Args:
            domain: The knowledge domain.
            task: The ML task type.
            complexity: One of 'demo', 'prototype', 'production'.

        Returns:
            Recommended number of records.
        """
        base = {
            Domain.TRAFFIC: {"demo": 500, "prototype": 5000, "production": 50000},
            Domain.MEDICAL: {"demo": 200, "prototype": 2000, "production": 20000},
            Domain.FINANCE: {"demo": 500, "prototype": 5000, "production": 50000},
            Domain.NLP: {"demo": 100, "prototype": 1000, "production": 10000},
            Domain.IOT: {"demo": 500, "prototype": 5000, "production": 100000},
            Domain.ECOMMERCE: {"demo": 500, "prototype": 5000, "production": 50000},
        }

        multipliers = {
            TaskType.TIME_SERIES_FORECAST: 1.5,
            TaskType.RAG_RETRIEVAL: 0.5,
            TaskType.CODE_TESTING: 0.2,
        }

        records = base.get(domain, {"demo": 300, "prototype": 3000, "production": 30000})
        size = records.get(complexity, 3000)
        size = int(size * multipliers.get(task, 1.0))
        return max(10, size)
