"""
Dataset Generator — 高质量数据集生成框架

A professional toolkit for generating high-quality, domain-specific datasets
for machine learning, RAG systems, and research prototyping.
"""

__version__ = "1.0.0"
__author__ = "Dataset Generator Contributors"
__license__ = "MIT"

from dataset_generator.core.config import DatasetConfig, Domain
from dataset_generator.core.schema import FieldSchema, DatasetSchema
from dataset_generator.core.types import FieldType, TaskType, OutputFormat

__all__ = [
    "DatasetConfig",
    "Domain",
    "FieldSchema",
    "DatasetSchema",
    "FieldType",
    "TaskType",
    "OutputFormat",
]
