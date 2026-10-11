# US Elections Forecast

Predicting 2026 US House midterm outcomes using a logistic regression model trained on historical House results (1976–2024).

**Current forecast (Oct 2026):** 220 Dem / 214 Rep median, 63.9% chance of Democratic majority.

## Project structure

```
app.py                  Streamlit frontend
predict.py              CLI — runs full pipeline and logs results
src/
  features.py           Data loading, feature engineering
  model.py              Model training, walk-forward CV, Monte Carlo simulation
  forecast.py           Backend called by the Streamlit app
  logging_config.py     Log setup (console + timestamped file)
data/                   Raw data files (not in repo — see below)
notebooks/
  explore.ipynb         EDA and pipeline walkthrough
tests/                  Pytest test suite (40 tests)
```

## Model overview

Features used:
- `dem_lean` — rolling 4-election average Dem vote share for the district
- `dem_lean_trend` — linear trend slope over the same window
- `pres_lean` — average Dem presidential vote share across 2020 + 2024 (by district)
- `pres_approval` / `pres_disapproval` — sign-adjusted incumbent approval ratings
- `generic_ballot` — final pre-election Dem – Rep polling margin
- `is_incumbent_on_ballot` — whether the sitting representative is running
- `redistricting` — flag for post-census redistricting years
- `nation_incumbent_party` / `district_incumbent_party` — one-hot encoded

Monte Carlo simulation uses correlated errors: a national shock shared across all districts plus an independent district-level shock, calibrated to historical walk-forward seat error (~15 seats MAE).

## Data

Place all files in the `data/` folder.

**MIT Election Data and Science Lab — U.S. House 1976–2024**
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
Download `1976-2024-house.tab`.

**Presidential approval ratings (American Presidency Project)**
https://docs.google.com/spreadsheets/d/1iEl565M1mICTubTtoxXMdxzaHzAcPTnb3kpRndsrfyY/edit?gid=1517284305#gid=1517284305
Download as `Approval-Ratings-for-POTUS-raw.xls`.

**Presidential results by congressional district (The Downballot)**
Two files needed:
- 2008–2020: https://docs.google.com/spreadsheets/d/1XVzQpDSQZtN3jCx3IiZwR3wsm-7kGqKKGJ1Q-Q3aJSk
- 2020–2024: https://docs.google.com/spreadsheets/d/1kFr-fMiMlOe2SAfrJkEeHVCHBXS2dGN9PXKv5UW5Fco
Download each as CSV and name them `president-2008-2020-by-district.csv` and `president-2020-2024-by-district.csv`.

## How to run

**Streamlit app (recommended)**

```
python -m streamlit run app.py
```

Opens at http://localhost:8501. Shows the seat distribution, competitive districts, and a per-district detail view.

**Command-line pipeline**

```
python predict.py
```

Logs the full walk-forward validation, feature importance stability, seat projection, and competitive districts. Output also saved to a timestamped file in `logs/`.

**Tests**

```
python -m pytest tests/
```

**Notebook**

Open `notebooks/explore.ipynb` for EDA and step-by-step pipeline walkthrough.
