# Data dictionary

## Raw source files (`data/raw/`, `data/interim/`)

### `1976-2024-house.tab` (MIT Election Lab, `data/raw/`)

Tab-separated. One row per candidate per district-year. Key columns (see `Col` in
[src/constants.py](../src/constants.py)):

| Column | Meaning |
|---|---|
| `year` | Election year |
| `state_po` | Two-letter state postal code |
| `district` | District number (0 = at-large) |
| `party` | Candidate's party, e.g. `DEMOCRAT`, `REPUBLICAN` (see `Val` below for the full value set, including historical fusion-ticket labels) |
| `candidate` | Candidate name |
| `candidatevotes` | Votes received by this candidate |
| `totalvotes` | Total votes cast in this district-year |
| `stage` | Election stage, e.g. `GEN` (general) |
| `special`, `runoff`, `writein`, `unofficial`, `fusion_ticket` | Boolean-ish flags for non-standard races |
| `state_fips`, `state_cen`, `state_ic` | Alternate state identifiers (FIPS / Census / ICPSR) — unused by the pipeline, kept for cross-referencing against other datasets |
| `office`, `mode`, `version` | Source-file bookkeeping columns, not used by the pipeline |

### `Approval-Ratings-for-POTUS-raw.xls` (`data/interim/`)

Built by `src/data/build_approval_history.py` from 9 pooled UCSB per-president pages — one row
per sampled poll.

| Column | Meaning |
|---|---|
| `start`, `end` | Poll fielding start/end dates |
| Approve % | Presidential approval, named per-source (normalized on load — see `load_approval` in `src/data/loaders.py`) |
| Disapprove % | Presidential disapproval |
| `unsure` | "Unsure"/"No data" share — present under several raw spellings across the 9 source pages, normalized to one name |

`build_dataset()` selects, for each House election year, the poll closest to (but before) that
election.

### `president-2008-2020-by-district.csv`, `president-2020-2024-by-district.csv` (`data/raw/`)

Built by `src/data/build_pres_lean_data.py` from The Downballot's public sheets — one row per
congressional district per presidential election year, with each candidate's vote share as a
column (used to compute presidential lean, `Col.PRES_LEAN`).

## `Col` — engineered column names (`src/constants.py`)

Columns created while building the feature matrix (`src/features/features.py`), beyond the raw
columns above:

| `Col` member | String value | Meaning |
|---|---|---|
| `DEM_VOTES` / `REP_VOTES` | `dem_votes` / `rep_votes` | Votes for the Democratic / Republican candidate in a district-year |
| `DEM_VOTE_SHARE` | `dem_vote_share` | Dem share of the two-party vote |
| `DEM_LEAN` | `dem_lean` | District's Dem lean relative to the national environment |
| `DEM_LEAN_TREND` | `dem_lean_trend` | Change in `dem_lean` from the prior cycle |
| `VOTE_SHARE` | `vote_share` | Generic vote-share column used in intermediate aggregation |
| `PRES_APPROVAL` / `PRES_DISAPPROVAL` | `pres_approval` / `pres_disapproval` | Presidential approval/disapproval at time of election, from the approval file |
| `PRES_PARTY` | `pres_party` | Sitting president's party for that House election year |
| `GENERIC_BALLOT` | `generic_ballot` | Net Dem advantage (Dem% − Rep%) in final generic-ballot polling (see `GENERIC_BALLOT` dict below) |
| `PREV_WINNER` | `prev_winner` | Winning party of this district in the prior cycle |
| `CANDIDATES` | `candidates` | Number of candidates on the ballot |
| `IS_INCUMBENT` | `is_incumbent_on_ballot` | Whether an incumbent is running |
| `REDISTRICTING` | `redistricting` | Whether this cycle followed redistricting (see `REDISTRICTING_YEARS`) |
| `NATION_INCUMBENT` | `nation_incumbent_party` | Incumbent party at the national (presidential) level |
| `DISTRICT_INCUMBENT` | `district_incumbent_party` | Incumbent party in this specific district |
| `DEM_WON` | `dem_won` | Target label: did the Democrat win this district-year |
| `PRES_LEAN` | `pres_lean` | District's presidential lean, from the two presidential-by-district files |
| `FUNDRAISING_DEM_SHARE` | `fundraising_dem_share` | Dem share of total district fundraising |
| `LARGE_DONOR_DEM_SHARE` | `large_donor_dem_share` | Dem share of large-donor (itemized) contributions |
| `PAC_DEM_SHARE` | `pac_dem_share` | Dem share of PAC contributions |
| `SPENDING_DEM_SHARE` | `spending_dem_share` | Dem share of campaign spending |
| `SURPLUS_DEM_SHARE` | `surplus_dem_share` | Dem share of cash-on-hand surplus |
| `PREV_DEM_SHARE` | `prev_dem_vote_share` | Dem vote share in the prior cycle for this district |
| `INC_X_BALLOT` | `inc_x_ballot` | Interaction term: incumbency × generic ballot |
| `ECON_GROWTH` | `econ_growth` | Real GDP growth (see `ECON_GROWTH` dict below) |
| `UNEMPLOYMENT` | `unemployment` | Unemployment rate (see `UNEMPLOYMENT` dict below) |
| `COOK_PROB` | `cook_prob` | Cook Political Report-derived probability, where available |

## `Val` — categorical values (`src/constants.py`)

The `party` column's value set, including historical ballot-line oddities that the loaders need
to recognize rather than drop: `GEN`, `DEMOCRAT`, `REPUBLICAN`, `INDEPENDENT`,
`DEMOCRATIC-FARMER-LABOR` (Minnesota's Dem-aligned party), `FOGLIETTA (DEMOCRAT)` (one
historical fusion-ticket label), `INDEPENDENT-REPUBLICAN`.

## Named constants and lookup dictionaries (`src/constants.py`)

| Name | Meaning |
|---|---|
| `DEM_MAJORITY_SEATS` | Seats needed for a House majority (218) |
| `COOK_ALPHA` | Blend weight given to Cook Political Report ratings (0.35, an informed prior — see the constant's own comment for why this isn't empirically fit) |
| `COMPETITIVE_MARGIN` | A district counts as "competitive" when `\|proba_dem - 0.5\| < COMPETITIVE_MARGIN` |
| `PRES_PARTY` | Year → sitting president's party, 1976-2026 |
| `GENERIC_BALLOT` | Year → net Dem advantage in final pre-election generic-ballot polling (Gallup/RCP/FiveThirtyEight final averages — see in-file source comments per year) |
| `ECON_GROWTH` | Year → real GDP growth, Q3 YoY % (FRED `GDPC1`) |
| `UNEMPLOYMENT` | Year → unemployment rate, % (FRED `UNRATE`) |
| `REDISTRICTING_YEARS` | Set of House election years following decennial redistricting |
| `PRES_YEAR_FOR_HOUSE` | House election year → which presidential election's results to use for `pres_lean` |
| `OPEN_SEATS_2026` | Set of `(state_po, district)` pairs with no incumbent running in 2026 |

## Forecast output fields (used by `app.py` / `api.py`, not in `Col`)

Produced by `run_forecast()` (`src/models/forecast.py`) and consumed by the Streamlit app, the
API, and the RAG/narrative/agent modules:

| Field | Meaning |
|---|---|
| `label` | Display label for a district, e.g. `"PA-7"` |
| `proba_dem` | Model's predicted probability the Democrat wins this district |
| `dem_lean` | District's Dem lean (see `Col.DEM_LEAN` above) |
| `is_open` | Whether this is an open seat (no incumbent) in 2026 — see `OPEN_SEATS_2026` |
| `district_incumbent` | Incumbent party for this district, 2026 — renamed from `Col.PARTY` in `run_forecast()` (not the same string as `Col.DISTRICT_INCUMBENT`, which is the engineered training-data column) |
| `seats_sim` | Array of simulated Democratic seat counts (one per Monte Carlo draw) |
| `n_districts` | Number of districts in the simulation |
| `districts` | The full per-district forecast DataFrame (contains all of the above, per row) |
