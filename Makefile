.PHONY: install install-dev download-data train infer eval lint test clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

download-data:
	python scripts/download_data.py

train:
	python scripts/train.py

infer:
	python scripts/infer.py

eval:
	python scripts/eval.py

lint:
	ruff check src/ scripts/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=ai_model_engineering_lab --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info/ .eggs/ __pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
