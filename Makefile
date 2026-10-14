.PHONY: help app test typecheck analyze predict tune rag cook db indicators agent calibrate-sigma freeze install install-dev data

PYTHON = python

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install          Install runtime dependencies + this package (editable)"
	@echo "  install-dev      Also install dev tooling (mypy, stubs)"
	@echo "  data             Check required data files, fetch what's safely automatable"
	@echo "  app              Start the Streamlit app"
	@echo "  test             Run the test suite"
	@echo "  typecheck        Run mypy over src/ and api.py"
	@echo "  analyze          Run all 6 diagnostic scripts (logs to logs/analysis/)"
	@echo "  predict          Train model + MLflow logging + 2026 forecast"
	@echo "  tune             Optuna hyperparameter tuning for LightGBM"
	@echo "  calibrate-sigma  Calibrate Monte Carlo sigma_national from walk-forward residuals"
	@echo "  rag              Build the ChromaDB RAG index from current forecast"
	@echo "  cook             Parse Cook Political Report ratings"
	@echo "  db               Build the SQLite forecast database"
	@echo "  indicators       Fetch latest economic indicators"
	@echo "  agent            Run the LLM forecast agent (requires GROQ_API_KEY)"
	@echo "  freeze           Snapshot the current environment to requirements-lock.txt"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

data:
	$(PYTHON) scripts/fetch_data.py

typecheck:
	$(PYTHON) -m mypy src/ api.py

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt

app:
	streamlit run app.py

test:
	$(PYTHON) -m pytest tests/ -q --tb=short

analyze:
	$(PYTHON) scripts/run_analysis.py

predict:
	$(PYTHON) scripts/predict.py

tune:
	$(PYTHON) scripts/tune.py

calibrate-sigma:
	$(PYTHON) scripts/calibrate_sigma.py

rag:
	$(PYTHON) scripts/build_rag_index.py

cook:
	$(PYTHON) scripts/parse_cook.py

db:
	$(PYTHON) scripts/build_db.py

indicators:
	$(PYTHON) scripts/update_indicators.py

agent:
	$(PYTHON) scripts/forecast_agent.py
