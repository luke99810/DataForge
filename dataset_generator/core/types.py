"""Type definitions for dataset generation."""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    """Supported knowledge domains."""

    TRAFFIC = "traffic"
    MEDICAL = "medical"
    FINANCE = "finance"
    NLP = "nlp"
    IOT = "iot"
    ECOMMERCE = "ecommerce"
    CUSTOM = "custom"

    @classmethod
    def from_label(cls, label: str) -> "Domain":
        """Infer domain from a natural language label."""
        mapping = {
            "交通": cls.TRAFFIC,
            "traffic": cls.TRAFFIC,
            "transport": cls.TRAFFIC,
            "医疗": cls.MEDICAL,
            "medical": cls.MEDICAL,
            "healthcare": cls.MEDICAL,
            "金融": cls.FINANCE,
            "finance": cls.FINANCE,
            "风控": cls.FINANCE,
            "nlp": cls.NLP,
            "自然语言": cls.NLP,
            "文本": cls.NLP,
            "text": cls.NLP,
            "物联网": cls.IOT,
            "iot": cls.IOT,
            "传感器": cls.IOT,
            "sensor": cls.IOT,
            "电商": cls.ECOMMERCE,
            "ecommerce": cls.ECOMMERCE,
            "零售": cls.ECOMMERCE,
            "retail": cls.ECOMMERCE,
        }
        return mapping.get(label.lower(), cls.CUSTOM)


class TaskType(str, Enum):
    """ML/DL task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECAST = "time_series_forecast"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    GENERATION = "generation"
    RAG_RETRIEVAL = "rag_retrieval"
    CODE_TESTING = "code_testing"

    @classmethod
    def from_label(cls, label: str) -> "TaskType":
        mapping = {
            "分类": cls.CLASSIFICATION,
            "classification": cls.CLASSIFICATION,
            "回归": cls.REGRESSION,
            "regression": cls.REGRESSION,
            "预测": cls.TIME_SERIES_FORECAST,
            "forecast": cls.TIME_SERIES_FORECAST,
            "时序": cls.TIME_SERIES_FORECAST,
            "time series": cls.TIME_SERIES_FORECAST,
            "聚类": cls.CLUSTERING,
            "clustering": cls.CLUSTERING,
            "异常检测": cls.ANOMALY_DETECTION,
            "anomaly": cls.ANOMALY_DETECTION,
            "生成": cls.GENERATION,
            "generation": cls.GENERATION,
            "rag": cls.RAG_RETRIEVAL,
            "检索": cls.RAG_RETRIEVAL,
            "retrieval": cls.RAG_RETRIEVAL,
            "测试": cls.CODE_TESTING,
            "testing": cls.CODE_TESTING,
        }
        return mapping.get(label.lower(), cls.CLASSIFICATION)


class FieldType(str, Enum):
    """Data field types."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    JSON = "json"
    GEO = "geo"


class OutputFormat(str, Enum):
    """Supported output formats."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    EXCEL = "xlsx"
    SQL = "sql"
    MARKDOWN = "md"
