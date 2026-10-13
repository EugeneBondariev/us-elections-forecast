.PHONY: help app test analyze predict tune rag cook db indicators agent

PYTHON = python

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  app         Start the Streamlit app"
	@echo "  test        Run the test suite"
	@echo "  analyze     Run all 6 diagnostic scripts (logs to logs/analysis/)"
	@echo "  predict     Train model + MLflow logging + 2026 forecast"
	@echo "  tune        Optuna hyperparameter tuning for LightGBM"
	@echo "  rag         Build the ChromaDB RAG index from current forecast"
	@echo "  cook        Parse Cook Political Report ratings"
	@echo "  db          Build the SQLite forecast database"
	@echo "  indicators  Fetch latest economic indicators"
	@echo "  agent       Run the LLM forecast agent (requires ANTHROPIC_API_KEY)"

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
