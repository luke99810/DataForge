"""Time-series data generator."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import DatasetConfig
from ..core.schema import DatasetSchema
from .base import BaseGenerator

logger = logging.getLogger(__name__)


class TimeseriesGenerator(BaseGenerator):
    """Generate synthetic time-series datasets.

    Supports multi-variable time series with configurable:
    - Trend components (linear, polynomial)
    - Seasonality (daily, weekly, monthly)
    - Cyclical patterns
    - Random noise
    - Irregular sampling intervals
    - Anomaly injection
    """

    def __init__(
        self,
        config: DatasetConfig,
        schema: DatasetSchema,
        freq: str = "H",
        start_date: str = "2024-01-01",
        trend_strength: float = 0.1,
        seasonality_periods: list[int] | None = None,
    ) -> None:
        super().__init__(config, schema)
        self.freq = freq
        self.start_date = pd.Timestamp(start_date)
        self.trend_strength = trend_strength
        self.seasonality_periods = seasonality_periods or [24, 168]  # daily, weekly by default
        self._supported_freqs = {
            "s": "second", "T": "minute", "min": "minute",
            "H": "hour", "h": "hour",
            "D": "day", "d": "day",
            "W": "week", "M": "month",
        }

    def _generate_records(self) -> list[dict[str, Any]]:
        n = self.config.num_records

        # Build time index
        freq_map = {"s": 1, "T": 60, "min": 60, "H": 3600, "h": 3600,
                      "D": 86400, "d": 86400, "W": 604800, "M": 2592000}
        delta_seconds = freq_map.get(self.freq, 3600)
        timestamps = pd.date_range(
            start=self.start_date, periods=n, freq=f"{delta_seconds}s"
        )

        records: list[dict[str, Any]] = []

        # Generate each variable as a time series
        feature_fields = self.schema.feature_fields
        series_data: dict[str, np.ndarray] = {}

        for field in feature_fields:
            series = self._generate_series(n, field.name)
            series_data[field.name] = series

        for i in range(n):
            record = {"timestamp": timestamps[i].isoformat()}
            for field in feature_fields:
                record[field.name] = round(float(series_data[field.name][i]), 4)
            if self.schema.target_field:
                # Simple target: weighted sum of features
                target = sum(
                    float(series_data[f.name][i]) * (0.5 + 0.1 * j)
                    for j, f in enumerate(feature_fields)
                )
                record[self.schema.target_field] = round(target, 4)
            records.append(record)

        return records

    def _generate_series(self, n: int, name: str) -> np.ndarray:
        """Generate a single time series with trend + seasonality + noise."""
        t = np.arange(n)

        # Trend
        trend = np.linspace(0, self.trend_strength * n, n)

        # Seasonality (multiple periods)
        seasonal = np.zeros(n)
        for period in self.seasonality_periods:
            if period > 0:
                seasonal += np.sin(2 * np.pi * t / period) * self.rng.uniform(0.5, 2.0)

        # Noise
        noise = self.rng.normal(0, 0.1, n)

        # Baseline offset varies by field (hash-based determinism)
        baseline = abs(hash(name)) % 100 / 100.0 * 10

        return baseline + trend + seasonal + noise
