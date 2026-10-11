# US House 2026 Election Forecast

Predicts 2026 US House election outcomes district-by-district using a stacking ensemble
(Logistic Regression + LightGBM + Random Forest) with Monte Carlo simulation for seat-count uncertainty.

## Project structure

```
src/           Model pipeline, feature engineering, loaders, forecast backend, RAG, narrative
scripts/       One-off analysis scripts (Cook parser, SHAP, calibration, tuning, agent CLI, etc.)
tests/         Pytest suite — unit tests run without data files; integration tests skip if absent
api.py         FastAPI endpoints (cached on startup, /health, /refresh)
app.py         Streamlit dashboard
predict.py     CLI: walk-forward validation, MLflow logging, LLM narrative
```

## Data requirements

Place these files in `data/` (not committed):

| File | Source |
|---|---|
| `1976-2024-house.tab` | [MIT Election Lab — U.S. House 1976–2024](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2) |
| `Approval-Ratings-for-POTUS-raw.xls` | [American Presidency Project](https://docs.google.com/spreadsheets/d/1iEl565M1mICTubTtoxXMdxzaHzAcPTnb3kpRndsrfyY) |
| `president-2008-2020-by-district.csv` | [The Downballot 2008–2020](https://docs.google.com/spreadsheets/d/1XVzQpDSQZtN3jCx3IiZwR3wsm-7kGqKKGJ1Q-Q3aJSk) |
| `president-2020-2024-by-district.csv` | [The Downballot 2020–2024](https://docs.google.com/spreadsheets/d/1kFr-fMiMlOe2SAfrJkEeHVCHBXS2dGN9PXKv5UW5Fco) |

Optional: `cook.html` (run `python scripts/parse_cook.py` after saving) and `fec_fundraising.csv`.

## Quickstart

```bash
pip install -r requirements.txt
python predict.py          # CLI forecast + MLflow logging
streamlit run app.py       # dashboard at localhost:8501
uvicorn api:app --reload   # REST API at localhost:8000
docker compose up          # both services via Docker
python -m pytest tests/ -q # test suite
```

Set `ANTHROPIC_API_KEY` to enable LLM narrative generation and the agent CLI (`scripts/forecast_agent.py`).

## Model

Walk-forward CV (always train on years < test year) feeds a stacking ensemble — LR, LightGBM, and Random Forest with an LR meta-learner. Seat-count uncertainty comes from Monte Carlo simulation with correlated national shocks plus independent district shocks. When Cook Political Report ratings are available, a blending step (35% weight) adjusts district probabilities toward the Cook rating.

## Scripts

| Script | What it does |
|---|---|
| `parse_cook.py` | Cook Political Report HTML → `data/cook.csv` |
| `build_db.py` | Source CSVs → SQLite (`data/elections.db`) |
| `tune.py` | Optuna hyperparameter search for LightGBM |
| `pca_analysis.py` | PCA on feature matrix, plots loadings |
| `feature_significance.py` | t-test, chi-square, permutation tests per feature |
| `shap_analysis.py` | Per-district SHAP explanations |
| `calibration_analysis.py` | Reliability diagram: raw vs. calibrated LR |
| `learning_curves.py` | Accuracy vs. training size for LR and LightGBM |
| `build_rag_index.py` | Build ChromaDB index of district fact sheets |
| `forecast_agent.py` | Interactive LLM agent with tool use (requires API key) |
