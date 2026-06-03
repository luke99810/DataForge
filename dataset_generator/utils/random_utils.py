"""Seeded random utilities for reproducible data generation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class SeededRandom:
    """Deterministic random number generation with seed management.

    Usage:
        rand = SeededRandom(42)
        value = rand.uniform(0, 1)
        next_value = rand.integer(0, 100)
    """

    def __init__(self, seed: int = 42) -> None:
        self._base_seed = seed
        self._counter = 0
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the random state."""
        if seed is not None:
            self._base_seed = seed
        self._counter = 0
        self._rng = np.random.default_rng(self._base_seed)

    def next_seed(self) -> int:
        """Generate a deterministic sub-seed for the next operation."""
        self._counter += 1
        seed_bytes = f"{self._base_seed}:{self._counter}".encode()
        return int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16) % (2**31)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: int = 1) -> Any:
        """Generate uniform random float(s)."""
        values = self._rng.uniform(low, high, size=size)
        return values[0] if size == 1 else values

    def integer(self, low: int, high: int, size: int = 1) -> Any:
        """Generate random integer(s) in [low, high]."""
        values = self._rng.integers(low, high + 1, size=size)
        return int(values[0]) if size == 1 else values

    def choice(self, items: list[Any], size: int = 1, p: list[float] | None = None) -> Any:
        """Pick random element(s) from a list."""
        values = self._rng.choice(items, size=size, p=p)
        return values[0] if size == 1 else values.tolist()

    def normal(self, mean: float = 0, std: float = 1, size: int = 1) -> Any:
        """Generate normally distributed random float(s)."""
        values = self._rng.normal(mean, std, size=size)
        return float(values[0]) if size == 1 else values
