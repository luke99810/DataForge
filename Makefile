.PHONY: help install install-dev test lint format type-check clean build

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install with dev dependencies
	pip install -e ".[dev]"

install-all: ## Install with all optional dependencies
	pip install -e ".[all]"

test: ## Run the test suite
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=dataset_generator --cov-report=term-missing --cov-report=html

lint: ## Run ruff linter
	ruff check dataset_generator/ tests/ scripts/

lint-fix: ## Auto-fix linting issues
	ruff check dataset_generator/ tests/ scripts/ --fix

format: ## Format code with ruff
	ruff format dataset_generator/ tests/ scripts/

type-check: ## Run mypy type checker
	mypy dataset_generator/

check: ## Run all checks (lint + type-check + test)
	ruff check dataset_generator/ tests/ scripts/
	mypy dataset_generator/
	pytest tests/ -v

clean: ## Remove build artifacts and cache
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: ## Build distribution packages
	python -m build

generate-example: ## Generate example datasets for all domains
	mkdir -p examples/
	dataset-gen generate --domain traffic --num-records 1000 --output examples/traffic_sample.csv
	dataset-gen generate --domain medical --num-records 500  --output examples/medical_sample.csv
	dataset-gen generate --domain finance --num-records 1000 --output examples/finance_sample.csv
	@echo "Example datasets generated in examples/"
