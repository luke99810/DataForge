# Contributing to {{PROJECT_NAME}}

Thank you for your interest in contributing!

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Run `ruff check .` before committing

## Adding New Domains

1. Add to `dataset_generator/core/types.py` Domain enum
2. Add template in `dataset_generator/templates/__init__.py`
3. Add test in `tests/test_core.py`
