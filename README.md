# US House 2026 Election Forecast

Predicts 2026 US House election outcomes district-by-district using a stacking ensemble
(Logistic Regression + LightGBM + Random Forest) with Monte Carlo simulation for seat-count uncertainty.

## Project structure

```
src/           Model pipeline, feature engineering, loaders, forecast backend, RAG, narrative
scripts/       Pipeline and analysis scripts (predict, Cook parser, SHAP, calibration, tuning, agent CLI, etc.)
tests/         Pytest suite — unit tests run without data files; integration tests skip if absent
api.py         FastAPI endpoints (cached on startup, /health, /refresh)
app.py         Streamlit dashboard
```

## Getting the data

`data/` is not committed. Follow these steps in order to reproduce it from scratch.

### Step 1 — Required files (download manually)

| File | Where to get it |
|---|---|
| `data/1976-2024-house.tab` | [MIT Election Lab — U.S. House 1976–2024](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2) → Download ZIP → extract the `.tab` file |
| `data/Approval-Ratings-for-POTUS-raw.xls` | [American Presidency Project](https://docs.google.com/spreadsheets/d/1iEl565M1mICTubTtoxXMdxzaHzAcPTnb3kpRndsrfyY) → File → Download → Microsoft Excel |
| `data/president-2008-2020-by-district.csv` | [The Downballot 2008–2020](https://docs.google.com/spreadsheets/d/1XVzQpDSQZtN3jCx3IiZwR3wsm-7kGqKKGJ1Q-Q3aJSk) → File → Download → CSV |
| `data/president-2020-2024-by-district.csv` | [The Downballot 2020–2024](https://docs.google.com/spreadsheets/d/1kFr-fMiMlOe2SAfrJkEeHVCHBXS2dGN9PXKv5UW5Fco) → File → Download → CSV |

### Step 2 — Fundraising data (FEC API)

Requires a free API key from [fred.stlouisfed.org](https://api.data.gov/signup).

```bash
export FEC_API_KEY=your_key_here
python -c "
import os, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from src.fundraising import fetch_and_cache
fetch_and_cache('data/fundraising.csv', os.environ['FEC_API_KEY'])
"
```

This fetches 2014–2026 House fundraising totals and saves to `data/fundraising.csv`. Takes ~2 minutes.

### Step 3 — Cook Political Report ratings (optional)

Improves 2026 district probability estimates by blending in Cook ratings at 35% weight.

1. Save the Cook Political Report House ratings page as `data/cook.html` (File → Save Page As in your browser)
2. Run the parser:
```bash
python scripts/parse_cook.py
```
This produces `data/cook.csv`.

### Step 4 — Verify live indicators

Checks that `UNEMPLOYMENT[2026]`, `ECON_GROWTH[2026]` in `src/constants.py` match current FRED data.
Requires a free API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

```bash
export FRED_API_KEY=your_key_here
python scripts/update_indicators.py
```

Update `src/constants.py` manually if any values are flagged.

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/predict.py       # CLI forecast + MLflow logging
python scripts/run_analysis.py  # full diagnostic pipeline → outputs/
streamlit run app.py            # dashboard at localhost:8501
uvicorn api:app --reload        # REST API at localhost:8000
python -m pytest tests/ -q      # test suite
```

Set `ANTHROPIC_API_KEY` to enable LLM narrative generation and the agent CLI (`scripts/forecast_agent.py`).

## Model

Walk-forward CV (always train on years < test year) feeds a stacking ensemble — LR, LightGBM, and Random Forest with an LR meta-learner. Seat-count uncertainty comes from Monte Carlo simulation with correlated national shocks plus independent district shocks. When Cook Political Report ratings are available, a blending step (35% weight) adjusts district probabilities toward the Cook rating.

## Scripts

| Script | What it does |
|---|---|
| `predict.py` | CLI forecast: walk-forward validation, MLflow logging, LLM narrative |
| `run_analysis.py` | Run all 6 diagnostic scripts in sequence, outputs land in `outputs/` |
| `parse_cook.py` | Cook Political Report HTML → `data/cook.csv` |
| `build_db.py` | Source CSVs → SQLite (`data/elections.db`) |
| `tune.py` | Optuna hyperparameter search for LightGBM |
| `update_indicators.py` | Check live unemployment and GDP against hardcoded constants |
| `pca_analysis.py` | PCA on feature matrix, plots loadings |
| `feature_significance.py` | t-test, chi-square, permutation tests per feature |
| `shap_analysis.py` | Per-district SHAP explanations |
| `calibration_analysis.py` | Reliability diagram: raw vs. calibrated LR |
| `learning_curves.py` | Accuracy vs. training size for LR and LightGBM |
| `build_rag_index.py` | Build ChromaDB index of district fact sheets |
| `forecast_agent.py` | Interactive LLM agent with tool use (requires API key) |
