"""Base generator abstract class and shared utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import DatasetConfig
from ..core.schema import DatasetSchema

logger = logging.getLogger(__name__)


def _ensure_seed(config: DatasetConfig) -> np.random.Generator:
    """Create a seeded NumPy random generator from config."""
    return np.random.default_rng(config.seed)


class BaseGenerator(ABC):
    """Abstract base class for all data generators.

    Subclasses must implement `_generate_records()` which returns a list
    of dicts. The base class handles seeding, noise injection, missing
    value simulation, and output formatting.
    """

    def __init__(self, config: DatasetConfig, schema: DatasetSchema) -> None:
        self.config = config
        self.schema = schema
        self.rng = _ensure_seed(config)
        logger.info(
            "Initializing %s: %d records, seed=%d, noise=%.2f, missing=%.2f",
            self.__class__.__name__,
            config.num_records,
            config.seed,
            config.noise_ratio,
            config.missing_rate,
        )

    def generate(self) -> pd.DataFrame:
        """Generate the complete dataset and return as a DataFrame.

        The pipeline:
        1. Call _generate_records() for raw data
        2. Convert to DataFrame
        3. Inject noise if configured
        4. Inject missing values if configured
        5. Validate basic constraints
        """
        logger.info("Generating %d records...", self.config.num_records)
        records = self._generate_records()

        df = pd.DataFrame(records)

        if self.config.noise_ratio > 0:
            df = self._inject_noise(df)

        if self.config.missing_rate > 0:
            df = self._inject_missing(df)

        self._validate_dataframe(df)
        logger.info("Generation complete: %d rows x %d columns", len(df), len(df.columns))
        return df

    @abstractmethod
    def _generate_records(self) -> list[dict[str, Any]]:
        """Generate raw records. Must be implemented by subclasses."""
        ...

    def _inject_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject random noise into numeric columns."""
        noise_mask = self.rng.random(len(df)) < self.config.noise_ratio
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            if col == self.schema.target_field:
                continue
            std = df[col].std()
            if pd.isna(std) or std == 0:
                continue
            noise = self.rng.normal(0, std * 0.1, size=len(df))
            df.loc[noise_mask, col] = df.loc[noise_mask, col] + noise[noise_mask]
        return df

    def _inject_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject missing values based on missing_rate."""
        n_cells = len(df) * len(df.columns)
        n_missing = int(n_cells * self.config.missing_rate * 0.5)  # target ~half the rate as cells

        for _ in range(n_missing):
            row_idx = self.rng.integers(0, len(df))
            col_idx = self.rng.integers(0, len(df.columns))
            col = df.columns[col_idx]
            if col == self.schema.target_field:
                continue
            df.iloc[row_idx, col_idx] = np.nan

        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Basic validation after generation."""
        if len(df) != self.config.num_records:
            logger.warning(
                "Expected %d records but got %d", self.config.num_records, len(df)
            )

        expected_cols = set(self.schema.field_names)
        actual_cols = set(df.columns)
        extra_cols = actual_cols - expected_cols
        missing_cols = expected_cols - actual_cols

        if extra_cols:
            logger.info("Extra columns generated: %s", extra_cols)
        if missing_cols:
            logger.warning("Missing expected columns: %s", missing_cols)
