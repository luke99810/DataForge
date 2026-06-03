#!/usr/bin/env python3
"""Generate datasets from schema files (JSON/YAML).

Delegates to dataset_generator.cli:from-schema.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_generator.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = ["dataset-gen", "from-schema", "--help"]
    elif sys.argv[1] not in ("from-schema",):
        sys.argv.insert(1, "from-schema")
    sys.exit(main())
