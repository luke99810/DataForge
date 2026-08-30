# Contributing to Dataset Generator

Thank you for your interest in contributing! This document outlines the process for contributing code, documentation, and domain templates.

---

## Development Setup

```bash
# Clone and install
git clone https://github.com/luke99810/DataForge.git
cd dataset-generator
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .

# Type check
mypy dataset_generator/
```

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with 100-character line length
- Use type hints on all public functions and methods
- Write Google-style docstrings for public APIs
- Run `ruff check .` before committing
- Run `mypy dataset_generator/` to verify types

---

## Pull Request Process

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write tests** for any new functionality:
   ```bash
   pytest tests/ -v --cov=dataset_generator
   ```

3. **Update documentation** if your change affects public APIs or adds new features.

4. **Ensure all checks pass** — CI runs linting, type checking, and tests automatically.

5. **Submit a PR** with a clear description of:
   - What problem it solves
   - How you tested it
   - Any breaking changes

---

## Adding a New Domain

1. Add the domain to `dataset_generator/core/types.py` (Domain enum):
   ```python
   class Domain(Enum):
       ...
       NEW_DOMAIN = "new_domain"
   ```

2. Create the template in `dataset_generator/templates/__init__.py`:
   ```python
   _NEW_DOMAIN_TEMPLATE = [
       FieldSchema(name="feature_1", dtype="float64", description="..."),
       FieldSchema(name="feature_2", dtype="category", description="..."),
   ]
   ```

3. Add a test in `tests/test_core.py`:
   ```python
   def test_new_domain_template():
       fields = get_domain_fields(Domain.NEW_DOMAIN)
       assert len(fields) >= 2
   ```

4. Update this `CONTRIBUTING.md` and `README.md` if applicable.

---

## Adding a New Generator

1. Create a new file in `dataset_generator/generators/`:
   ```python
   from dataset_generator.generators.base import BaseGenerator

   class MyGenerator(BaseGenerator):
       def _generate_records(self) -> list[dict]:
           # Implement generation logic
           ...
   ```

2. Register it in `dataset_generator/generators/__init__.py`

3. Add a CLI subcommand in `dataset_generator/cli.py`

4. Add tests in `tests/test_generators.py`

---

## Reporting Issues

Use the [Issue Templates](.github/ISSUE_TEMPLATE/) for bug reports and feature requests.

---

## Community

- Be respectful and constructive
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Review others' PRs with care
