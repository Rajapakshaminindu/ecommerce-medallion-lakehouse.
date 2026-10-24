.PHONY: help install pipeline pipeline-fast score test lint app clean

help:
	@echo "E-Commerce Medallion Lakehouse & ML Pipeline"
	@echo "Available commands:"
	@echo "  make install       Install all project dependencies"
	@echo "  make pipeline      Run full ELT pipeline (Default: 300 customers, 2000 orders)"
	@echo "  make pipeline-fast Run quick ELT pipeline smoke run (50 customers, 300 orders)"
	@echo "  make score         Execute batch churn & LTV scoring and export retention list"
	@echo "  make test          Run all unit tests"
	@echo "  make lint          Run Flake8 linting checks"
	@echo "  make app           Launch interactive Streamlit dashboard"
	@echo "  make clean         Clean bytecode, cache files, and logs"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install flake8 pytest

pipeline:
	python src/run_pipeline.py

pipeline-fast:
	python src/run_pipeline.py --customers 50 --orders 300 --seed 42

score:
	python src/predict_batch.py --threshold 0.50 --top-at-risk 10

test:
	python -m unittest discover tests -v

lint:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

app:
	streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
