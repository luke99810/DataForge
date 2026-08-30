# DataForge

> Desktop agent workstation for synthetic dataset construction — from literature or schema to validated, gated, provenance-tracked training and evaluation data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/luke99810/DataForge/actions/workflows/ci.yml/badge.svg)](https://github.com/luke99810/DataForge/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-354%20passed-3fb950.svg)](EVALUATION.md)

**DataForge** produces domain-specific, research-grade synthetic datasets for machine learning, reinforcement learning, LLM fine-tuning, and agent evaluation. It combines a deterministic generation kernel (`dataset_generator`) with an orchestration layer (`dataforge`) that adds content-addressed provenance, fingerprint-based caching, quality gates, and an LLM-driven planning loop.

The full lifecycle is covered: feature discovery from academic literature → schema construction → data generation → quality validation → gated export with a data card.

---

## Architecture

Dependencies are strictly downward. The kernel has no knowledge of the orchestration layer and never invokes a language model.

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **L4** Interface | `dataforge.app`, `dataforge.mcp_server` | Desktop application, HTTP API, MCP server, CLI |
| **L3** Orchestration | `dataforge.harness`, `dataforge.planner` | PEVR loop, budget control, plateau detection, gate scoring |
| **L2** Capabilities | `dataforge.nodes`, `dataforge.quality`, `dataforge.sources` | Node implementations, leakage and dedup analysis, external source resolution |
| **L1** Kernel | `dataset_generator` | Deterministic generation, schema system, validators |
| **L0** Provenance | `dataforge.core`, `dataforge.graph` | Artifact store, fingerprinting, lineage, DAG execution |

Only L3 is permitted to call a language model. Model output is treated as a proposal; retry, convergence, and threshold decisions are computed in code.

---

## Features

### Generation

- **Domain-Aware Generation** — Built-in templates for traffic, medical, finance, NLP, IoT, and e-commerce
- **Literature-Driven Features** — Extract input/output variables from uploaded PDFs, with mandatory `source_span` citation and a human confirmation gate
- **Schema-Driven Generation** — Define fields in JSON/YAML; deterministic and reproducible under a fixed seed
- **Time Series Support** — Trend, seasonality, cyclicity, and noise injection
- **RAG Datasets** — Documents, QA pairs, and multi-turn conversations with controllable complexity
- **RL Trajectories** — `(s, a, r, s', done)` episodes for `gridworld` and `inventory` environments, verified by dynamics replay
- **LLM Training Sets** — Instruction, preference, and multi-turn formats with dedup and contamination screening
- **Agent Evaluation Sets** — Programmatically checkable tasks across five categories and three difficulty tiers
- **Source-Aligned Generation** — Synthesize datasets isomorphic to a referenced benchmark's schema, without downloading its contents

### Orchestration

- **Content-Addressed Artifacts** — DataFrames hashed by content, dtype, and column order; Parquet-backed store
- **Fingerprint Caching** — Node fingerprint derives from upstream hashes, parameters, `code_version`, and seed; invalidation propagates by content
- **Lineage Reconstruction** — Any run is replayable from its trace JSONL, including node fingerprints and artifact hashes
- **Quality Gates** — Blocking thresholds on row count, completeness, distribution shift, and quality score; threshold relaxation is recorded in the data card
- **Refine Loop** — Bounded Plan–Execute–Verify–Refine cycle with five terminal states (`SUCCESS`, `PLATEAU`, `BLOCKED`, `EXHAUSTED`, `NEEDS_HUMAN`), all determined in code
- **Multi-Format Output** — CSV, JSON, Parquet, Excel, JSONL

### Validation

- **Target Leakage Detection** — Correlation and mutual-information screening, with a domain whitelist to suppress physically expected relationships
- **Temporal Split Auditing** — Detects random splits applied to time-ordered data
- **Deduplication** — Exact duplicates removed automatically; near-duplicates flagged for review rather than merged
- **Benchmark Contamination Screening** — Prompt-level and full-text comparison against a local benchmark registry
- **Document Extraction** — Parse PDF, Word, and Markdown to auto-detect data patterns

---

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[app]"    # Desktop application (FastAPI, uvicorn, pywebview)
pip install -e ".[llm]"    # LLM planning and paper extraction (openai, pymupdf)
pip install -e ".[mcp]"    # MCP server
pip install -e ".[dev]"    # pytest, ruff, mypy
```

The kernel requires only `numpy`, `pandas`, `pyyaml`, `openpyxl`, and `pyarrow`. No GPU is required by any component.

---

## Quick Start

### Desktop Application

```bash
python -m dataforge.app.desktop                          # Native window
python -m dataforge.app.desktop --no-window --port 8000  # Server only
```

Five views: **Configure** (field editor, paper upload, variable-card confirmation, gate thresholds, live trace), **Graph** (executed DAG, nodes colored by status), **Quality** (gate detail, leakage report, distributions), **Data** (200-row preview), **History**.

### CLI

```bash
# Demonstrate caching and invalidation propagation
python -m dataforge demo

# Generate RL / agent-evaluation / LLM training data
python -m dataforge gen rl         -n 200 --env gridworld
python -m dataforge gen agent-eval -n 40
python -m dataforge gen llm        -n 500

# Generate a dataset isomorphic to a referenced benchmark
python -m dataforge align gsm8k -n 500
python -m dataforge align list

# Evaluation-set discrimination sweep and Inspect export
python -m dataforge evalsuite -n 40 --export ./out

# Inspect run history and lineage
python -m dataforge runs
python -m dataforge explain <run_id>
```

The kernel CLI remains available for direct generation:

```bash
dataset-gen generate --domain traffic --num-records 5000 --format csv
dataset-gen from-schema --schema schemas/traffic.yaml --output data/
dataset-gen rag --num-docs 100 --num-qa 500 --output data/rag/
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

### MCP Server

```bash
python -m dataforge.mcp_server
```

Exposes 19 tools, including `df_plan`, `df_generate`, `df_validate`, `df_analyze`, `df_export`, `df_parse_paper`, `df_dataset_from_paper`, `df_rl_dataset`, `df_llm_dataset`, `df_agent_eval_set`, `df_eval_discrimination`, `df_export_inspect`, `df_align_to_source`, `df_inspect_source`, `df_list_benchmarks`, `df_lineage`, `df_list_runs`, `df_fetch_artifact`, and `df_run_assertions`.

Two constraints apply to MCP callers: `df_generate` does not accept gate thresholds, and `df_dataset_from_paper` requires a non-empty `adopt` list, preserving the human confirmation gate.

---

## Pipelines

### Schema-Driven

```
schema ──> generate ──┬──> gate ──> export
                      └──> analyze
```

### Paper-Driven

```
parse ──> extract ──> confirm ──> schema ──> generate ──┬──> gate ──> export
(rules)   (LLM)      (human)      (code)                └──> analyze
```

Three constraints govern extraction:

- Related-work sections are excluded from the prompt. Variables the model attributes to related work are unchecked by default.
- `source_span` is mandatory. Variables without a locatable span in the source document are discarded.
- The `confirm` node is a blocking gate and cannot be bypassed from any interface.

Resumption after confirmation is a consequence of fingerprint caching: the confirmation result is submitted as a `confirm` node parameter, upstream fingerprints are unchanged, and only `confirm` and its downstream nodes recompute. The `extract` node — the only billed step — is served from cache.

### Source-Aligned

```
source ──> schema ──> generate ──> gate ──> export
```

Resolves a benchmark name, `hf://datasets/<org>/<name>`, or GitHub repository URL to a field structure, then generates an isomorphic synthetic dataset. Only metadata endpoints are contacted; dataset bodies are never downloaded. Network access is opt-in via `--network`.

---

## Quality Gates

| Threshold | Default | Effect when violated |
|-----------|---------|----------------------|
| `min_rows` | 50 | Block |
| `min_completeness` | 0.90 | Block |
| `max_ks` | 0.10 | Block |
| `min_quality_score` | 80.0 | Block |

Additional gates cover schema conformance, target leakage, and unit conflicts.

When a gate blocks, the `export` node reports `SKIPPED` and no artifact is written. Thresholds may be relaxed, but every deviation from the default is recorded in the data card alongside the original value. The `synthetic: true` marking applied at export cannot be disabled.

### Validation Dimensions

Every generated dataset passes through four validation dimensions:

- **Completeness** — missing value ratio, row completeness
- **Consistency** — type integrity, logical constraints
- **Distribution** — statistical summaries, class balance
- **Format** — schema compliance, encoding, delimiter detection

Results are output as a structured Markdown or JSON report.

---

## Agent Evaluation Suite

Tasks are generated across five categories — `compute`, `constrained`, `file_ops`, `safety`, `structured_output` — at three difficulty tiers (`easy`, `medium`, `hard`).

Correctness is determined by a declarative assertion set. Ten assertion kinds are supported:

`file_exists` · `file_contains` · `file_json_path` · `value_equals` · `value_close` · `value_in` · `tool_called` · `tool_not_called` · `output_matches` · `no_exception`

The DSL deliberately excludes `python`, `eval`, and `shell` assertions. Evaluation sets are designed to be distributed, and arbitrary code execution in a shared artifact is a remote-execution vector.

### Discrimination Sweep

`dataforge evalsuite` runs five behavior-defined simulated agents (`perfect`, `sloppy`, `lazy`, `wrong`, `noop`) against the generated task set and reports pass rates by difficulty and category. The spread between the highest and lowest profile quantifies whether the assertion set separates capability levels. Measured spread on a 40-task set is 1.00.

### Inspect Export

```bash
python -m dataforge evalsuite -n 40 --export ./out
```

Produces `tasks.jsonl` (one Inspect `Sample` per line), `eval_task.py` (an Inspect `@task` with a DataForge-backed scorer), and `README.md`. Inspect supplies model execution, sandboxing, and logging; grading remains with the DataForge assertion runner. Model-graded scorers are not used.

---

## External Sources

Network access is restricted to a hardcoded host whitelist. Adding a source requires an explicit code change.

| Host | Purpose |
|------|---------|
| `datasets-server.huggingface.co`, `huggingface.co` | Dataset schema metadata |
| `api.github.com`, `raw.githubusercontent.com` | Repository structure |
| `arxiv.org`, `export.arxiv.org` | Paper metadata |

A local benchmark registry (`gsm8k`, `mmlu`, `humaneval`, `imdb`, `swe_bench`) carries a `registry_version` stamp and resolves without network access. Unrecognized identifiers produce an explicit error rather than an inferred structure.

---

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATAFORGE_API_KEY` | LLM credential for planning and paper extraction | — |
| `DATAFORGE_MODEL` | Model identifier | `deepseek-chat` |
| `DATAFORGE_BASE_URL` | API endpoint | `https://api.deepseek.com/v1` |
| `OPENAI_API_KEY` | Fallback credential | — |

Credentials are read from the environment only. They are never written to configuration files, never enter node parameters, and therefore never reach a fingerprint or trace record. When no credential is configured, LLM-dependent nodes raise an explicit error rather than falling back to rule-based extraction.

---

## Project Structure

```
DataForge/
├── dataset_generator/          # L1 — deterministic kernel
│   ├── core/                   # Types, config, schema system
│   ├── generators/             # Tabular, time series, RAG generators
│   ├── extractors/             # Document pattern extraction
│   ├── validators/             # Quality checks & reporting
│   ├── templates/              # Domain-specific field templates
│   └── utils/                  # I/O, random seeding, logging
├── dataforge/                  # L0/L2/L3/L4 — orchestration layer
│   ├── core/                   # Artifact store, fingerprint, lineage, trace
│   ├── graph/                  # DAG definition and executor
│   ├── harness/                # Loop, budget, gates, scoring, refine
│   ├── nodes/                  # Node implementations
│   ├── quality/                # Leakage and dedup analysis
│   ├── sources/                # Benchmark registry, HF/GitHub resolution
│   ├── llm/                    # Client and output contracts
│   ├── app/                    # Desktop shell, HTTP API, web assets
│   └── mcp_server.py           # MCP server
├── docs/eval/                  # Evaluation charts and sweep data
├── scripts/                    # Thin CLI wrapper scripts
├── tests/                      # pytest test suite
├── references/                 # Domain knowledge & usage guides
├── assets/github/              # Dataset project templates
├── EVALUATION.md               # Measured evaluation report
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

## Testing

```bash
python -m pytest tests/ -q
```

354 tests across 19 files: 17 kernel tests and 337 orchestration-layer tests. All LLM-dependent paths are exercised against a stub client, so the suite is deterministic and requires no network access or credentials.

---

## Evaluation

[EVALUATION.md](EVALUATION.md) reports measured results for cache behavior, invalidation precision, injected-defect detection, deduplication, and evaluation-set discrimination, together with the known limitations of each measurement.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setup, code style, and PR process.

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure process.

## License

MIT — see [LICENSE](LICENSE) for full text.

## Citation

```bibtex
@software{dataforge_2026,
  title = {DataForge: Provenance-tracked synthetic dataset construction for ML, RL, and agent evaluation},
  year = {2026},
  url = {https://github.com/luke99810/DataForge},
}
```
