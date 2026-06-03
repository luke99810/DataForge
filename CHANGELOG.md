# Changelog

All notable changes to Dataset Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-03

### Added

- Core Python package (`dataset_generator`) with modular architecture
- `core/` subpackage: type system (`DatasetConfig`, enums), schema system (`FieldSchema`, `DatasetSchema`)
- `generators/` subpackage: `BaseGenerator` abstract class + `TabularGenerator`, `TimeseriesGenerator`, `RAGGenerator`, `SchemaDrivenGenerator`
- `extractors/` subpackage: `DocumentExtractor` with 6 detection strategies (Markdown tables, key-value, CSV headers, etc.)
- `validators/` subpackage: four-dimensional quality checks (completeness, consistency, distribution, format) + Markdown/JSON reports
- `templates/` subpackage: 6 built-in domain templates (Traffic, Medical, Finance, NLP, IoT, E-Commerce)
- `utils/` subpackage: `SeededRandom`, I/O helpers, structured logging
- Unified CLI (`dataset-gen` / `dg`) with 5 subcommands: `generate`, `from-schema`, `from-doc`, `rag`, `validate`
- `pyproject.toml` with setuptools build, optional dev/parquet dependencies, pytest/ruff/mypy config
- `tests/` directory with `test_core.py` and `test_generators.py`
- Reference documentation: domain templates, dataset formats, literature search guide
- GitHub project asset templates (`assets/github/`)
- WorkBuddy skill definition (`SKILL.md`)

### Notes

- Initial release supporting 6 domains with auto-estimated data volume
- Noise injection and missing-value simulation at the base generator level
- Deterministic generation via seed parameter
- Type hints throughout, mypy-compatible
