# US House 2026 Election Forecast

Predicts 2026 US House election outcomes district-by-district using a stacking ensemble
(Logistic Regression + LightGBM + Random Forest) with Monte Carlo simulation for
seat-count uncertainty.

**Current forecast (Oct 2026):** Run `python predict.py` to see current projections.

---

## Project structure

```
src/           Model pipeline, feature engineering, data loaders, forecast backend
scripts/       One-off analysis scripts (Cook parser, SHAP, calibration, tuning, etc.)
tests/         Pytest suite (tests skip automatically if data files are absent)
api.py         FastAPI endpoints
app.py         Streamlit dashboard
predict.py     CLI training, walk-forward validation, and MLflow logging
```

---

## Data requirements

Place all files in the `data/` directory. This directory is not committed to the repo.

| File | Source |
|---|---|
| `1976-2024-house.tab` | [MIT Election Lab — U.S. House 1976–2024](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2) |
| `Approval-Ratings-for-POTUS-raw.xls` | [American Presidency Project approval ratings](https://docs.google.com/spreadsheets/d/1iEl565M1mICTubTtoxXMdxzaHzAcPTnb3kpRndsrfyY) |
| `president-2008-2020-by-district.csv` | [The Downballot — 2008–2020 presidential results by district](https://docs.google.com/spreadsheets/d/1XVzQpDSQZtN3jCx3IiZwR3wsm-7kGqKKGJ1Q-Q3aJSk) |
| `president-2020-2024-by-district.csv` | [The Downballot — 2020–2024 presidential results by district](https://docs.google.com/spreadsheets/d/1kFr-fMiMlOe2SAfrJkEeHVCHBXS2dGN9PXKv5UW5Fco) |

**Optional files:**

- `cook.html` — Save the Cook Political Report ratings page locally, then run
  `python scripts/parse_cook.py` to generate `data/cook.csv`. Enables Cook blending in
  the forecast.
- `fec_fundraising.csv` — FEC candidate fundraising data. Used as an optional feature.

---

## How to run

**Install dependencies**

```bash
pip install -r requirements.txt
```

**CLI forecast**

```bash
python predict.py
```

Runs walk-forward validation, prints feature importance, outputs district projections,
and logs results to MLflow and a timestamped file in `logs/`.

**Streamlit dashboard**

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Shows the seat distribution, competitive districts, and
a per-district detail view.

**FastAPI**

```bash
uvicorn api:app --reload
```

**Docker**

```bash
docker compose up
```

**Tests**

```bash
python -m pytest tests/ -q
```

Tests skip automatically when the required data files are not present.

---

## Model overview

The pipeline uses walk-forward cross-validation: the model is always trained on election
years strictly before the test year, which mirrors a real forecasting scenario and avoids
look-ahead bias. The final model is a stacking ensemble of Logistic Regression,
LightGBM, and Random Forest with a Logistic Regression meta-learner. Seat-count
uncertainty is estimated via Monte Carlo simulation that applies correlated national
shocks (shared across all districts) plus independent district-level shocks, calibrated
to historical walk-forward seat error. When Cook Political Report ratings are available,
a blending step adjusts district win probabilities toward the Cook rating.

---

## Key results

Run `python predict.py` to see current projections. Results are logged to MLflow
(`mlflow ui`).

---

## Scripts

| Script | Description |
|---|---|
| `parse_cook.py` | Parses a locally saved Cook Political Report HTML ratings page and writes `data/cook.csv` |
| `build_db.py` | One-time loader: imports source CSVs into a local SQLite database (`data/elections.db`) |
| `tune.py` | Optuna hyperparameter search for LightGBM using walk-forward CV |
| `pca_analysis.py` | PCA on the feature matrix — plots explained variance and component loadings |
| `feature_significance.py` | Statistical hypothesis tests (t-test, chi-square, permutation) for each feature |
| `shap_analysis.py` | Computes and saves per-district SHAP feature explanations |
| `calibration_analysis.py` | Reliability diagram comparing raw vs. isotonic-calibrated model probabilities |
| `learning_curves.py` | Plots accuracy vs. training set size for Logistic Regression and LightGBM |

---

## Tests

```bash
python -m pytest tests/ -q
```

Tests require the data files listed above. Any test that depends on a missing file is
skipped automatically — you will see `s` markers in the pytest output rather than
failures.
