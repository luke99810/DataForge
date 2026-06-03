#!/usr/bin/env python3
"""Generate tabular datasets (CSV, JSON, Parquet, Excel).

Delegates to dataset_generator.cli:generate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_generator.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = ["dataset-gen", "generate"]
    elif sys.argv[1] not in ("generate",):
        sys.argv.insert(1, "generate")
    sys.exit(main())
