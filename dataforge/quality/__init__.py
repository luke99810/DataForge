"""质量检查。"""

from .dedup import (
    DedupReport, check_contamination, dedup_indices, find_duplicates, jaccard,
)

from .leakage import LeakageFinding, check_all, check_split_overlap, check_target_leakage, check_temporal_leakage, has_error

__all__ = [
    "DedupReport", "check_contamination", "dedup_indices",
    "find_duplicates", "jaccard",
    "LeakageFinding", "check_all", "check_split_overlap",
    "check_target_leakage", "check_temporal_leakage", "has_error",
]
