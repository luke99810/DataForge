# Dataset Generator

> High-quality synthetic dataset generation toolkit for ML, deep learning, RAG, and research prototyping.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/luke99810/Dataset-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/luke99810/Dataset-Generator/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Dataset Generator** is a professional toolkit that produces domain-specific, research-grade synthetic datasets. Whether you are training ML models, building RAG pipelines, or prototyping algorithms, it handles the full lifecycle: feature discovery from academic literature → schema construction → data generation → quality validation → GitHub-ready output.

---

## Features

- **Domain-Aware Generation** — Built-in templates for traffic, medical, finance, NLP, IoT, and e-commerce
- **Literature-Driven Features** — Search academic papers to determine input/output variables before generating data
- **Multi-Format Output** — CSV, JSON, Parquet, Excel, JSONL
- **RAG-Ready Datasets** — Documents + QA pairs + multi-turn conversations with controllable complexity
- **Time Series Support** — Trend, seasonality, cyclicity, and noise injection
- **Schema-Driven Generation** — Define fields in JSON/YAML; deterministic, reproducible generation
- **Quality Validation** — Completeness, consistency, distribution, and format checks with Markdown reports
- **Document Extraction** — Parse PDF, Word, and Markdown files to auto-detect data patterns
- **GitHub-Ready Projects** — Auto-generate README, data dictionary, schema docs, and reproduction scripts

---

## Quick Start

### Installation

```bash
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

### CLI Usage

```bash
# Generate tabular data for a domain
dataset-gen generate --domain traffic --num-records 5000 --format csv

# Generate from a schema file
dataset-gen from-schema --schema schemas/traffic.yaml --output data/

# Build a RAG dataset
dataset-gen rag --num-docs 100 --num-qa 500 --output data/rag/

# Validate an existing dataset
dataset-gen validate --input data/traffic.csv --report validation_report.md
```

### Python API

```python
from dataset_generator import DatasetConfig, generators
from dataset_generator.core.types import Domain, TaskType, OutputFormat

config = DatasetConfig(
    domain=Domain.TRAFFIC,
    task_type=TaskType.PREDICTION,
    num_records=5000,
    output_format=OutputFormat.CSV,
    seed=42,
)

gen = generators.TabularGenerator(config)
df = gen.generate()
print(f"Generated {len(df)} records with {len(df.columns)} features")
```

---

## Project Structure

```
dataset-generator/
├── dataset_generator/          # Core Python package
│   ├── core/                   # Types, config, schema system
│   ├── generators/             # Tabular, time series, RAG generators
│   ├── extractors/             # Document pattern extraction
│   ├── validators/             # Quality checks & reporting
│   ├── templates/              # Domain-specific field templates
│   └── utils/                  # I/O, random seeding, logging
├── scripts/                    # Thin CLI wrapper scripts
├── tests/                      # pytest test suite
├── references/                 # Domain knowledge & usage guides
├── assets/github/              # Dataset project templates
├── .github/                    # CI workflows & issue templates
├── pyproject.toml              # Build config & dependencies
├── Makefile                    # Convenience commands
└── SKILL.md                    # WorkBuddy skill definition
```

---

## Supported Domains

| Domain | Records (default) | Features | Task Types |
|--------|-------------------|----------|------------|
| Traffic | 5,000–50,000 | 11+ | Prediction, Classification, Clustering |
| Medical | 1,000–30,000 | 12+ | Classification, Survival Analysis |
| Finance | 3,000–60,000 | 10+ | Regression, Anomaly Detection |
| NLP | 500–10,000 | 8+ | Classification, Generation |
| IoT | 10,000–100,000 | 9+ | Regression, Anomaly Detection |
| E-Commerce | 5,000–100,000 | 10+ | Classification, Recommendation |

---

## Quality Validation

Every generated dataset passes through four validation dimensions:

- **Completeness** — missing value ratio, row completeness
- **Consistency** — type integrity, logical constraints
- **Distribution** — statistical summaries, class balance
- **Format** — schema compliance, encoding, delimiter detection

Results are output as a structured Markdown or JSON report.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setup, code style, and PR process.

## License

MIT — see [LICENSE](LICENSE) for full text.

## Citation

If you use Dataset Generator in your research, please cite:

```bibtex
@software{dataset_generator_2026,
  title = {Dataset Generator: High-quality synthetic dataset generation toolkit},
  year = {2026},
  url = {https://github.com/user/dataset-generator},
}
```
