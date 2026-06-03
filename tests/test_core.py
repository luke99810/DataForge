"""Core module tests."""

import pytest

from dataset_generator.core.config import DatasetConfig
from dataset_generator.core.schema import DatasetSchema, FieldConstraint, FieldSchema
from dataset_generator.core.types import Domain, FieldType, OutputFormat, TaskType


class TestDomain:
    def test_from_label_chinese(self):
        assert Domain.from_label("交通") == Domain.TRAFFIC
        assert Domain.from_label("医疗") == Domain.MEDICAL
        assert Domain.from_label("金融") == Domain.FINANCE

    def test_from_label_english(self):
        assert Domain.from_label("traffic") == Domain.TRAFFIC
        assert Domain.from_label("nlp") == Domain.NLP
        assert Domain.from_label("iot") == Domain.IOT

    def test_unknown_label(self):
        assert Domain.from_label("unknown_stuff") == Domain.CUSTOM


class TestTaskType:
    def test_from_label(self):
        assert TaskType.from_label("分类") == TaskType.CLASSIFICATION
        assert TaskType.from_label("regression") == TaskType.REGRESSION
        assert TaskType.from_label("预测") == TaskType.TIME_SERIES_FORECAST

    def test_unknown(self):
        assert TaskType.from_label("blah") == TaskType.CLASSIFICATION


class TestDatasetConfig:
    def test_default(self):
        c = DatasetConfig()
        assert c.num_records == 1000
        assert c.seed == 42
        assert c.noise_ratio == 0.0

    def test_invalid_noise(self):
        with pytest.raises(ValueError):
            DatasetConfig(noise_ratio=1.5)

    def test_invalid_language(self):
        with pytest.raises(ValueError):
            DatasetConfig(language="fr")

    def test_estimate_size(self):
        size = DatasetConfig.estimate_size(Domain.TRAFFIC, TaskType.REGRESSION, "demo")
        assert size >= 10


class TestFieldSchema:
    def test_basic(self):
        f = FieldSchema(name="age", field_type=FieldType.INTEGER, description="Age of person")
        d = f.to_dict()
        assert d["name"] == "age"
        assert d["type"] == "integer"

    def test_roundtrip(self):
        f = FieldSchema(
            name="price",
            field_type=FieldType.FLOAT,
            constraints=FieldConstraint(min=0.0, max=100.0, precision=2),
        )
        d = f.to_dict()
        f2 = FieldSchema.from_dict(d)
        assert f2.name == "price"
        assert f2.constraints.precision == 2


class TestDatasetSchema:
    def test_validate_no_duplicates(self):
        s = DatasetSchema(
            name="test",
            fields=[
                FieldSchema(name="a"),
                FieldSchema(name="a"),
            ],
        )
        issues = s.validate()
        assert len(issues) >= 1
        assert "Duplicate" in issues[0]

    def test_missing_target(self):
        s = DatasetSchema(
            name="test",
            target_field="not_here",
            fields=[FieldSchema(name="a")],
        )
        issues = s.validate()
        assert len(issues) >= 1
