"""Data generator tests."""

import pandas as pd

from dataset_generator.core.config import DatasetConfig
from dataset_generator.core.schema import DatasetSchema, FieldConstraint, FieldSchema
from dataset_generator.core.types import FieldType, TaskType
from dataset_generator.generators.tabular import TabularGenerator
from dataset_generator.generators.timeseries import TimeseriesGenerator


class TestTabularGenerator:
    def test_basic_generation(self):
        config = DatasetConfig(num_records=100, seed=42)
        schema = DatasetSchema(
            name="test",
            fields=[
                FieldSchema(name="id", field_type=FieldType.STRING, unique=True),
                FieldSchema(name="age", field_type=FieldType.INTEGER,
                            constraints=FieldConstraint(min=0, max=100)),
                FieldSchema(name="score", field_type=FieldType.FLOAT,
                            constraints=FieldConstraint(min=0.0, max=100.0)),
                FieldSchema(name="category", field_type=FieldType.CATEGORICAL,
                            constraints=FieldConstraint(categories=["A", "B", "C"])),
                FieldSchema(name="is_valid", field_type=FieldType.BOOLEAN),
            ],
        )

        gen = TabularGenerator(config, schema)
        df = gen.generate()

        assert len(df) == 100
        assert set(df.columns) == {"id", "age", "score", "category", "is_valid"}
        assert df["age"].between(0, 100).all()
        assert df["score"].between(0.0, 100.0).all()
        assert df["category"].isin(["A", "B", "C"]).all()
        assert df["is_valid"].isin([True, False]).all()

    def test_unique_constraint(self):
        config = DatasetConfig(num_records=50, seed=42)
        schema = DatasetSchema(
            name="test",
            fields=[
                FieldSchema(name="uid", field_type=FieldType.STRING, unique=True),
            ],
        )

        gen = TabularGenerator(config, schema)
        df = gen.generate()
        assert df["uid"].nunique() == 50

    def test_noise_injection(self):
        config = DatasetConfig(num_records=100, seed=42, noise_ratio=0.3)
        schema = DatasetSchema(
            name="test",
            fields=[
                FieldSchema(name="val", field_type=FieldType.FLOAT,
                            constraints=FieldConstraint(min=0.0, max=100.0)),
            ],
        )

        gen = TabularGenerator(config, schema)
        df = gen.generate()
        assert len(df) == 100


class TestTimeseriesGenerator:
    def test_basic_generation(self):
        config = DatasetConfig(num_records=168, seed=42, task_type=TaskType.TIME_SERIES_FORECAST)
        schema = DatasetSchema(
            name="test_ts",
            fields=[
                FieldSchema(name="timestamp", field_type=FieldType.DATETIME),
                FieldSchema(name="value_a", field_type=FieldType.FLOAT),
                FieldSchema(name="value_b", field_type=FieldType.FLOAT),
            ],
            target_field="value_a",
        )

        gen = TimeseriesGenerator(config, schema, freq="H")
        df = gen.generate()

        assert len(df) == 168
        assert "timestamp" in df.columns
        assert "value_a" in df.columns
        assert "value_b" in df.columns
