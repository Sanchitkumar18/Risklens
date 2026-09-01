# ── RiskLens developer tasks ────────────────────────────────
# Usage: `make <target>`. PY points at a 3.11+ interpreter for creating the venv.
PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help venv install test test-unit test-integration lint run \
        migrate migrate-down revision sample-data load-data compose-up compose-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	$(PY) -m venv $(VENV)

install: venv ## Install dependencies into the venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

test: ## Run the full test suite
	$(BIN)/pytest

test-unit: ## Run only fast unit tests
	$(BIN)/pytest -m unit

test-integration: ## Run only database integration tests
	$(BIN)/pytest -m integration

migrate: ## Apply all migrations (alembic upgrade head)
	$(BIN)/alembic upgrade head

migrate-down: ## Roll back the latest migration
	$(BIN)/alembic downgrade -1

revision: ## Autogenerate a migration: make revision M="message"
	$(BIN)/alembic revision --autogenerate -m "$(M)"

sample-data: ## Generate the synthetic sample CSV (data/sample/)
	$(BIN)/python -m scripts.generate_sample_data

load-data: ## Load the sample CSV into the database (needs DATABASE_URL + migrate)
	$(BIN)/python -m scripts.load_sample_data

lint: ## Lint with ruff (if installed)
	$(BIN)/python -m ruff check app tests || echo "ruff not installed; skipping"

run: ## Run the API locally with autoreload
	$(BIN)/uvicorn app.main:app --reload

dashboard: ## Run the Streamlit dashboard (API must be running)
	$(BIN)/streamlit run dashboard/app.py

demo: ## Run the full end-to-end demo (self-contained, prints every step)
	$(BIN)/python -m scripts.run_demo

compose-up: ## Start postgres + api via docker compose
	docker compose up --build

compose-down: ## Stop and remove compose services
	docker compose down

clean: ## Remove caches and the venv
	rm -rf $(VENV) .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
