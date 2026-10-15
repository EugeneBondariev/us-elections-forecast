.PHONY: help app test test-e2e typecheck analyze predict tune rag db indicators agent calibrate-sigma freeze install install-dev fetch validate push

PYTHON = python
export PYTHONUNBUFFERED := 1

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install runtime dependencies + this package (editable)"
	@echo "  install-dev      Also install dev tooling (mypy, stubs, playwright)"
	@echo "  freeze           Snapshot the current environment to requirements-lock.txt"
	@echo ""
	@echo "Quality:"
	@echo "  push             Run typecheck + test, then git push (only if both pass)"
	@echo "  typecheck        Run mypy over src/ and api.py"
	@echo "  test             Run the fast test suite (excludes e2e)"
	@echo "  test-e2e         Run the real-browser Streamlit/API smoke tests (slow, opt-in -"
	@echo "                   needs: pip install playwright && playwright install chromium)"
	@echo ""
	@echo "Data:"
	@echo "  fetch            Fetch all data files (downloads + builds + inline validation)"
	@echo "  validate         Re-validate all data files already on disk"
	@echo "  indicators       Fetch latest economic indicators"
	@echo "  db               Build the SQLite forecast database"
	@echo ""
	@echo "Model:"
	@echo "  predict          Train model + MLflow logging + 2026 forecast"
	@echo "  tune             Optuna hyperparameter tuning for LightGBM"
	@echo "  calibrate-sigma  Calibrate Monte Carlo sigma_national from walk-forward residuals"
	@echo ""
	@echo "Analysis & App:"
	@echo "  analyze          Run all 6 diagnostic scripts (logs to logs/analysis/)"
	@echo "  app              Start the Streamlit app"
	@echo "  rag              Build the ChromaDB RAG index from current forecast"
	@echo "  agent            Run the LLM forecast agent (requires GROQ_API_KEY)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

push: typecheck test
	git push

fetch:
	$(PYTHON) -m src.data.fetch_data

validate:
	$(PYTHON) -m src.data.validate_data

typecheck:
	$(PYTHON) -m mypy src/ api.py

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt

app:
	streamlit run app.py

test:
	$(PYTHON) -m pytest tests/ -q --tb=short

test-e2e:
	$(PYTHON) -m pytest tests/test_app_e2e.py -m e2e -v

analyze:
	$(PYTHON) -m src.visualization.run_analysis

predict:
	$(PYTHON) -m src.models.predict

tune:
	$(PYTHON) -m src.models.tune

calibrate-sigma:
	$(PYTHON) -m src.models.calibrate_sigma

rag:
	$(PYTHON) -m src.visualization.build_rag_index

db:
	$(PYTHON) -m src.data.build_db

indicators:
	$(PYTHON) -m src.data.update_indicators

agent:
	$(PYTHON) -m src.visualization.forecast_agent
