#!/usr/bin/env python3
"""Generate time-series datasets.

Delegates to dataset_generator.cli:generate with --task time_series_forecast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_generator.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = ["dataset-gen", "generate", "--task", "time_series_forecast"]
    elif all(a not in sys.argv for a in ("--task",)):
        sys.argv.insert(1, "--task")
        sys.argv.insert(2, "time_series_forecast")
    if sys.argv[1] not in ("generate",):
        sys.argv.insert(1, "generate")
    sys.exit(main())
