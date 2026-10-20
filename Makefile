.PHONY: help app mlflow test test-fast test-e2e coverage typecheck lint analyze predict tune rag db indicators generic-ballot open-seats census candidate-quality poll-aggregation pipeline agent calibrate-sigma freeze install install-dev fetch validate push

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
	@echo "  typecheck        Run mypy over src/, api.py and app.py"
	@echo "  lint             Run ruff linter"
	@echo "  test             Run the full test suite (excludes e2e)"
	@echo "  test-fast        Re-run only tests affected by file changes (pytest-testmon)"
	@echo "  coverage         Run tests with coverage; open htmlcov/index.html to explore"
	@echo "  test-e2e         Run the real-browser Streamlit/API smoke tests (slow, opt-in -"
	@echo "                   needs: pip install playwright && playwright install chromium)"
	@echo ""
	@echo "Data:"
	@echo "  fetch            Fetch all data files (downloads + builds + inline validation)"
	@echo "  validate         Re-validate all data files already on disk"
	@echo "  indicators       Fetch latest economic indicators from FRED"
	@echo "  generic-ballot   Fetch generic ballot history (FiveThirtyEight + Wikipedia)"
	@echo "  open-seats       Fetch 2026 open-seat list from Wikipedia"
	@echo "  census           Fetch ACS 5-year demographics from Census Bureau API"
	@echo "  candidate-quality  Score candidates from FEC history + TF-IDF NLP"
	@echo "  poll-aggregation   Aggregate generic ballot polls with house-effects correction"
	@echo "  db               Build the SQLite forecast database"
	@echo ""
	@echo "Model:"
	@echo "  pipeline         Full run: fetch all data + retrain model (run once, then make app)"
	@echo "  predict          Train model + MLflow logging + 2026 forecast"
	@echo "  tune             Optuna hyperparameter tuning for LightGBM"
	@echo "  calibrate-sigma  Calibrate Monte Carlo sigma_national from walk-forward residuals"
	@echo ""
	@echo "Analysis & App:"
	@echo "  analyze          Run all 6 diagnostic scripts (logs to logs/analysis/)"
	@echo "  app              Start the Streamlit app"
	@echo "  mlflow           Start the MLflow UI (models/mlflow.db)"
	@echo "  rag              Build the ChromaDB RAG index from current forecast"
	@echo "  agent            Run the LLM forecast agent (requires GROQ_API_KEY)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

pre-commit: lint typecheck test-fast

push: lint typecheck test
	git push

pipeline: fetch census poll-aggregation candidate-quality predict

fetch:
	$(PYTHON) -m src.data.fetch_data

validate:
	$(PYTHON) -m src.data.validate_data

typecheck:
	$(PYTHON) -m mypy src/ api.py app.py

lint:
	$(PYTHON) -m ruff check . --fix

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt

app:
	$(PYTHON) -m streamlit run app.py

mlflow:
	$(PYTHON) -c 'exec("import time,webbrowser,socket\nfor _ in range(30):\n  s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n  if s.connect_ex((\"127.0.0.1\",5000))==0:\n    s.close();break\n  s.close();time.sleep(0.5)\nwebbrowser.open(\"http://127.0.0.1:5000\")")' &
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///models/mlflow.db

coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -q
	@echo "HTML report: htmlcov/index.html"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ --testmon -v --tb=short

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
	$(PYTHON) -m src.data.build_indicators

generic-ballot:
	$(PYTHON) -m src.data.build_generic_ballot

open-seats:
	$(PYTHON) -m src.data.build_open_seats

census:
	$(PYTHON) -m src.data.census

candidate-quality:
	$(PYTHON) -m src.data.candidate_nlp

poll-aggregation:
	$(PYTHON) -m src.data.poll_aggregation

agent:
	$(PYTHON) -m src.visualization.forecast_agent
