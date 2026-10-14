# US House 2026 Election Forecast

Predicts 2026 US House election outcomes district-by-district using a stacking ensemble
(Logistic Regression + LightGBM + Random Forest) with Monte Carlo simulation for seat-count uncertainty.

## Project structure

```
src/            Cookiecutter Data Science layout — package root, importable as `src`:
  data/           Fetching/loading/building datasets (fetch_data, build_db, parse_cook, ...)
  features/       Feature engineering + feature-level analysis (build_dataset, PCA, t-tests, ...)
  models/         Training, forecasting, calibration, tuning (model.py, forecast.py, predict.py, ...)
  visualization/  Narrative generation, RAG, diagnostic plots, the LLM agent CLI
  constants.py, paths.py, logging_config.py   Cross-cutting, used by every subpackage above
tests/          Pytest suite, mirroring src/'s layout (tests/data/, tests/features/, ...).
                Unit tests (synthetic fixtures) run without data files; integration tests
                (the `real_data` fixture, from conftest.py) skip if the 4 required files are absent
docs/           Sphinx documentation scaffold
references/     Data dictionaries — what each column/constant means
api.py          FastAPI endpoints (cached on startup, /health, /refresh)
app.py          Streamlit dashboard

data/           Not committed (see below) — Cookiecutter Data Science layout:
  raw/            Primary source files, exactly as downloaded/manually assembled. Immutable —
                   never hand-edit; fix a mistake by re-fetching/re-copying, not in place.
  external/       Third-party supplementary data (Cook ratings page, raw FEC records). Same
                   immutability rule, just not the core dataset.
  interim/        First-pass transformations of raw/external data, not yet final features.
  processed/      Derived runtime artifacts (forecast cache, RAG index) — safe to delete.
models/         Not committed — trained/tuned model artifacts (best LightGBM params,
                 calibrated Monte Carlo sigma). Not "data" in the DS sense, so it's separate.
reports/        Not committed — generated analysis output (text/csv), regenerable.
  figures/        Generated plots/images from the diagnostic scripts, regenerable.
```

## Getting the data

`data/` and `models/` are not committed — see `src/paths.py` for what lives where and
why. Both are created automatically (empty) the moment any script, test, or the app/API
imports `src.paths`, so the directory structure exists before you've run anything,
`make` or not.

**`make data` (or `python -m src.data.fetch_data`) fetches all 4 required files
automatically** — no manual downloads, no browser, no signup needed in the common case.
It also automates the FEC fundraising fetch and Cook ratings parse when the right
key/file is already present, and reports clearly if any step still needs attention.
Each required file uses a different mechanism depending on its source's licensing (see
`src/data/fetch_data.py`'s module docstring for the full reasoning per file):

### Required files

| File | How it's fetched |
|---|---|
| `data/raw/1976-2024-house.tab` | Downloaded + SHA256-verified from a GitHub Release mirror on this repo. The primary source ([MIT Election Lab on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2)) requires a one-time Guestbook form, so this repo mirrors an already-obtained copy — confirmed CC0 1.0 (public domain) via Harvard Dataverse's own metadata API, so redistributing it is fine. You'd only need the primary source yourself if the mirror ever breaks. |
| `data/interim/Approval-Ratings-for-POTUS-raw.xls` | Fetched fresh every run by `src/data/build_approval_history.py`, from [presidency.ucsb.edu](https://www.presidency.ucsb.edu/statistics/data)'s 9 dedicated per-president pages (Ford through Biden) — picks the last poll before each House election. Lives in `interim/`, not `raw/`, despite the filename: it's a selection built from 9 pooled external sources, not a passthrough of anything downloaded untouched (the same shape of transformation as `cook.html` → `cook.csv`). **Never gets a static mirror**: this data is "adapted from the Gallup Poll" (a private company) and the page carries an explicit "Copyright © The American Presidency Project" notice; UCSB's Terms of Service permit only "one copy... for personal non-commercial home use," which fetching fresh each run (not redistributing a static copy) satisfies. |
| `data/raw/president-2008-2020-by-district.csv`, `president-2020-2024-by-district.csv` | Fetched fresh every run by `src/data/build_pres_lean_data.py`, from [The Downballot](https://www.the-downballot.com/p/data)'s public sheets. Their terms explicitly welcome reuse with attribution ("we welcome (almost) any use of our data as long as you cite us"), with one restriction — "please refrain from reproducing entire spreadsheets" — so, same reasoning as the approval file, this fetches fresh rather than mirroring a static copy. **Cite The Downballot** if you publish anything derived from this. |

All 4 land in `data/raw/` — treat them as immutable once fetched. If you find something wrong,
re-run the fetch rather than hand-editing the file in place; `load_approval`/`load_clean_house_data`
validate on load (see `src/data/validators.py`) and will raise with a specific error if something's off.

Only these 4 files are required — they're exactly what `conftest.py`'s `real_data` fixture checks for and what `build_dataset()` needs. Everything below is optional: the pipeline and app run fine without it, just with fewer features / a less-informed 2026 blend.

### Optional: fundraising data (FEC API)

Requires a free API key from [api.data.gov](https://api.data.gov/signup) (the FEC API authenticates through data.gov, not FRED).

```bash
export FEC_API_KEY=your_key_here
python -c "
import os, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from src.data.fundraising import fetch_and_cache
from src.paths import DataFile
fetch_and_cache(DataFile.FUNDRAISING, os.environ['FEC_API_KEY'])
"
```

This fetches 2014–2026 House fundraising totals, caches the raw per-candidate records to
`data/external/fundraising_raw.csv`, and saves the aggregated Dem-share features to
`data/interim/fundraising.csv`. Takes ~2 minutes.

### Optional: Cook Political Report ratings

Improves 2026 district probability estimates by blending in Cook ratings at 35% weight.

1. Save the Cook Political Report House ratings page as `data/external/cook.html` (File → Save Page As in your browser)
2. Run the parser:
```bash
python -m src.data.parse_cook
```
This produces `data/interim/cook.csv`.

### Optional: verify live indicators

Checks that `UNEMPLOYMENT[2026]`, `ECON_GROWTH[2026]` in `src/constants.py` match current FRED data.
Requires a free API key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).
This only prints a comparison to the console — it doesn't write any file or gate anything else.

```bash
export FRED_API_KEY=your_key_here
python -m src.data.update_indicators
```

Update `src/constants.py` manually if any values are flagged.

### SQLite mirror (optional, not currently used by the pipeline)

`src/data/build_db.py` loads the house/approval files into `data/processed/elections.db`, and
`src/data/database.py` has loader equivalents (`load_house_from_db`, `load_approval_from_db`) that read
from it. Nothing in `build_dataset()` calls them today — the pipeline always reads the raw files
directly — so skip this unless you specifically want to query the source data via SQLite.
`elections.db` is otherwise only used for forecast *result* persistence (`save_forecast`/
`load_latest_forecast`, used by `api.py`/`app.py`
to cache the last run), which is created automatically on first run regardless.

---

## Quickstart

### Prerequisites

- Python 3.11+ — [python.org/downloads](https://www.python.org/downloads/)
- Git

Nothing else is assumed. `make` is an optional shortcut covered at the bottom of this section —
every command below works without it.

### Setup

```bash
git clone <this repo's URL>
cd us-elections-forecast
python -m pip install -r requirements.txt
python -m pip install -e .
python -m src.data.fetch_data
```

> **If a bare `pip ...` command says "command not found"** even though Python is installed:
> use `python -m pip ...` instead, every time, in every command below. Some Python installs
> (common on Windows) don't generate a standalone `pip.exe` launcher, but the `pip` *module*
> always works via `python -m pip` regardless. This isn't a broken install — just use the
> `-m pip` form and move on.

`python -m pip install -e .` makes `src` importable from anywhere, rather than only when the
shell's working directory happens to be the repo root. `requirements.txt` pins most packages
exactly; the handful that don't aren't yet verified against a known-good install in this repo's
CI — run `python -m pip freeze > requirements-lock.txt` after installing to snapshot a
fully-resolved lockfile if you want one.

`python -m src.data.fetch_data` checks/fetches the data files — see "Getting the data" above for
exactly what's automatic versus manual.

### Run it

```bash
python -m src.models.predict         # CLI forecast + MLflow logging
python -m src.visualization.run_analysis  # full diagnostic pipeline -> reports/
streamlit run app.py            # dashboard at localhost:8501
uvicorn api:app --reload        # REST API at localhost:8000
python -m pytest tests/ -q      # test suite
python -m pip install -r requirements-dev.txt && python -m mypy src/ api.py   # type check
```

### Optional: `make` shortcuts

Everything above has a shorter `make` equivalent (`make install`, `make data`, `make test`, ...
— see `Makefile` for the full list). `make` isn't installed by default on Windows. To get it:

```powershell
winget install GnuWin32.Make
```

If that exact package ID doesn't resolve, run `winget search make` first to see what's actually
available on your machine and pick from the list. Either way, **open a new terminal window**
afterward — PATH changes don't apply to already-open shells — then `make --version` should work.

Set `GROQ_API_KEY` to enable LLM narrative generation, the Ask tab, and the agent CLI (`src/visualization/forecast_agent.py`).

## Model

Walk-forward CV (always train on years < test year) feeds a stacking ensemble — LR, LightGBM, and Random Forest with an LR meta-learner. Seat-count uncertainty comes from Monte Carlo simulation with correlated national shocks plus independent district shocks. When Cook Political Report ratings are available, a blending step (35% weight) adjusts district probabilities toward the Cook rating.

The Monte Carlo `sigma_national` parameter is calibrated (not hand-picked) by `src/models/calibrate_sigma.py`: it binary-searches for the value whose simulated seat-count spread matches the empirical spread of realized walk-forward national seat errors, then caches the result to `models/calibrated_sigma.json`. `sigma_district` and the Cook blend weight (`COOK_ALPHA`) are held as documented constants in `src/constants.py` — there isn't enough held-out district-level history (and no archived historical Cook ratings) to identify them empirically the same way.

## Pipeline scripts

Each lives inside the `src/<subpackage>/` it belongs to (no separate top-level `scripts/`
directory — see `src/paths.py`'s Cookiecutter Data Science docstring). Run any of them with
`python -m <module>`:

| Module | What it does |
|---|---|
| `src.data.fetch_data` | Fetches/checks all required and optional data files, running the modules below as needed |
| `src.data.build_approval_history` | Rebuilds `data/interim/Approval-Ratings-for-POTUS-raw.xls` from UCSB's per-president approval pages (local use only — never redistribute, see module docstring) |
| `src.data.build_pres_lean_data` | Rebuilds the two `president-*-by-district.csv` files from The Downballot's sheets (local use only — never redistribute, see module docstring) |
| `src.data.parse_cook` | Cook Political Report HTML (`data/external/cook.html`) → `data/interim/cook.csv` |
| `src.data.build_db` | Source CSVs → SQLite (`data/processed/elections.db`) |
| `src.data.update_indicators` | Check live unemployment and GDP against hardcoded constants |
| `src.features.feature_significance` | t-test, chi-square, permutation tests per feature → `reports/` |
| `src.features.ols_analysis` | OLS regression on structural features → `reports/` + `reports/figures/` |
| `src.features.pca_analysis` | PCA on feature matrix, plots loadings → `reports/figures/` |
| `src.models.predict` | CLI forecast: walk-forward validation, MLflow logging, LLM narrative |
| `src.models.tune` | Optuna hyperparameter search for LightGBM → `models/best_params_lgbm.json` |
| `src.models.calibrate_sigma` | Fits Monte Carlo `sigma_national` to walk-forward residuals → `models/calibrated_sigma.json` |
| `src.models.learning_curves` | Accuracy vs. training size for LR and LightGBM → `reports/figures/` |
| `src.visualization.run_analysis` | Run all 6 diagnostic modules in sequence, outputs land in `reports/` |
| `src.visualization.shap_analysis` | Per-district SHAP explanations → `reports/` + `reports/figures/` |
| `src.visualization.calibration_analysis` | Reliability diagram: raw vs. calibrated LR → `reports/figures/` |
| `src.visualization.build_rag_index` | Build ChromaDB index of district fact sheets |
| `src.visualization.forecast_agent` | Interactive LLM agent with tool use (requires API key) |

## Docs

A Sphinx scaffold lives in `docs/` (`python -m pip install -r requirements-dev.txt && make -C docs html`).
Column/constant definitions are documented in `references/data_dictionary.md`. `tox.ini` runs
the test suite in an isolated environment for CI-equivalent local checks (`tox`).

## License

MIT — see [LICENSE](LICENSE). This covers the code in this repository only. The *data* files
fetched into `data/` each carry their own source's licensing terms, which differ per file — see
"Getting the data" above for the specifics (CC0 for the house-results file; local-use-only
fetches for the approval ratings and presidential-lean files, which carry third-party terms that
restrict redistribution).
