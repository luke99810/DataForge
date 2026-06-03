#!/usr/bin/env python3
"""Generate RAG datasets with documents and QA pairs.

Delegates to dataset_generator.cli:rag.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_generator.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = ["dataset-gen", "rag", "--help"]
    elif sys.argv[1] not in ("rag",):
        sys.argv.insert(1, "rag")
    sys.exit(main())
