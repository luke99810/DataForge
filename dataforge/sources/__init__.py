"""外部数据源：只取元数据与结构，不下载数据本体。"""

from .base import (
    ALLOWED_HOSTS, CachedFetcher, Fetcher, HttpFetcher, SourceError, SourceRef,
    StubFetcher, identify,
)
from .registry import BENCHMARKS, SYNTHETIC_NOTICE, known, match
from .resolve import resolve

__all__ = [
    "ALLOWED_HOSTS", "BENCHMARKS", "CachedFetcher", "Fetcher", "HttpFetcher",
    "SYNTHETIC_NOTICE", "SourceError", "SourceRef", "StubFetcher", "identify",
    "known", "match", "resolve",
]
