from __future__ import annotations

import datetime
import warnings
from enum import StrEnum

__all__ = [
    "Col", "Val",
    "DEM_MAJORITY_SEATS", "COOK_ALPHA", "COMPETITIVE_MARGIN",
    "PRES_PARTY", "GENERIC_BALLOT", "ECON_GROWTH", "UNEMPLOYMENT",
    "REDISTRICTING_YEARS", "PRES_YEAR_FOR_HOUSE", "OPEN_SEATS_2026",
]


class Col(StrEnum):
    # raw source columns
    YEAR = "year"
    STATE = "state_po"
    DISTRICT = "district"
    PARTY = "party"
    CANDIDATE = "candidate"
    CANDIDATE_VOTES = "candidatevotes"
    TOTAL_VOTES = "totalvotes"
    STAGE = "stage"
    SPECIAL = "special"
    RUNOFF = "runoff"
    WRITE_IN = "writein"
    UNOFFICIAL = "unofficial"
    FUSION_TICKET = "fusion_ticket"
    STATE_FIPS = "state_fips"
    STATE_CEN = "state_cen"
    STATE_IC = "state_ic"
    OFFICE = "office"
    MODE = "mode"
    VERSION = "version"
    # approval source columns
    START = "start"
    END = "end"
    UNSURE = "unsure"
    # engineered columns
    DEM_VOTES = "dem_votes"
    REP_VOTES = "rep_votes"
    DEM_VOTE_SHARE = "dem_vote_share"
    DEM_LEAN = "dem_lean"
    DEM_LEAN_TREND = "dem_lean_trend"
    VOTE_SHARE = "vote_share"
    PRES_APPROVAL = "pres_approval"
    PRES_DISAPPROVAL = "pres_disapproval"
    PRES_PARTY = "pres_party"
    GENERIC_BALLOT = "generic_ballot"
    PREV_WINNER = "prev_winner"
    CANDIDATES = "candidates"
    IS_INCUMBENT = "is_incumbent_on_ballot"
    REDISTRICTING = "redistricting"
    NATION_INCUMBENT = "nation_incumbent_party"
    DISTRICT_INCUMBENT = "district_incumbent_party"
    DEM_WON = "dem_won"
    PRES_LEAN = "pres_lean"
    PRES_YEAR = "pres_year"
    FUNDRAISING_DEM_SHARE = "fundraising_dem_share"
    LARGE_DONOR_DEM_SHARE = "large_donor_dem_share"
    PAC_DEM_SHARE         = "pac_dem_share"
    SPENDING_DEM_SHARE    = "spending_dem_share"
    SURPLUS_DEM_SHARE     = "surplus_dem_share"
    PREV_DEM_SHARE = "prev_dem_vote_share"
    INC_X_BALLOT = "inc_x_ballot"
    ECON_GROWTH = "econ_growth"
    UNEMPLOYMENT = "unemployment"
    COOK_PROB = "cook_prob"


class Val(StrEnum):
    GEN = "GEN"
    DEMOCRAT = "DEMOCRAT"
    REPUBLICAN = "REPUBLICAN"
    INDEPENDENT = "INDEPENDENT"
    DEM_FARMER_LABOR = "DEMOCRATIC-FARMER-LABOR"
    FOGLIETTA_DEM = "FOGLIETTA (DEMOCRAT)"
    INDEPENDENT_REP = "INDEPENDENT-REPUBLICAN"


# --- Named constants (replaces inline magic numbers) ---

DEM_MAJORITY_SEATS = 218      # seats needed for a House majority

# Cook Political blend weight. Not empirically calibrated: that would require
# backtesting against archived historical Cook ratings district-by-district, which
# aren't available (only the current cycle's ratings are scraped). Treat this as an
# informed prior, not a fitted parameter — unlike sigma_national in src/model.py's
# calibrate_national_sigma(), which *is* fit from walk-forward residuals.
COOK_ALPHA = 0.35
COMPETITIVE_MARGIN = 0.15     # |proba_dem - 0.5| < this → competitive district


# --- Lookup dictionaries ---

PRES_PARTY = {
    1976: "REPUBLICAN",  # Ford
    1978: "DEMOCRAT",    # Carter
    1980: "DEMOCRAT",
    1982: "REPUBLICAN",  # Reagan
    1984: "REPUBLICAN",
    1986: "REPUBLICAN",
    1988: "REPUBLICAN",
    1990: "REPUBLICAN",  # Bush Sr
    1992: "REPUBLICAN",
    1994: "DEMOCRAT",    # Clinton
    1996: "DEMOCRAT",
    1998: "DEMOCRAT",
    2000: "DEMOCRAT",
    2002: "REPUBLICAN",  # Bush Jr
    2004: "REPUBLICAN",
    2006: "REPUBLICAN",
    2008: "REPUBLICAN",
    2010: "DEMOCRAT",    # Obama
    2012: "DEMOCRAT",
    2014: "DEMOCRAT",
    2016: "DEMOCRAT",
    2018: "REPUBLICAN",  # Trump
    2020: "REPUBLICAN",
    2022: "DEMOCRAT",    # Biden
    2024: "DEMOCRAT",
    2026: "REPUBLICAN",  # Trump
}

# Net Dem advantage in generic congressional ballot (Dem% - Rep%), final pre-election polling.
# Positive = Dem-favorable environment, negative = Rep-favorable.
# Sources: Gallup final likely-voter poll 1976-2000 (news.gallup.com/poll/21094),
#          RealClearPolitics final avg 2002-2016, FiveThirtyEight final avg 2018-2024.
# 1988: no Gallup final ballot on record; value estimated from adjacent context.
# 1980: Gallup RV showed D+15 (structurally biased); LV environment was Rep-favorable.
GENERIC_BALLOT = {
    1976: 14,   # Gallup LV
    1978: 9,    # Gallup
    1980: -5,   # Gallup LV (RV D+15 misleadingly large; actual House D+2.6)
    1982: 12,   # Gallup RV
    1984: 9,    # Gallup RV
    1986: 10,   # Gallup
    1988: 4,    # estimated — no Gallup final ballot conducted
    1990: 8,    # Gallup
    1992: 12,   # Gallup
    1994: -7,   # Gallup LV (RV showed D+4 — misleading; actual R+6.5)
    1996: 0,    # Gallup RV: tied
    1998: 1,    # Gallup LV ~0; RV showed D+7 (biased); actual D+1.8
    2000: 1,    # Gallup LV (RV showed D+8)
    2002: -2,   # RCP final avg R+1.7
    2004: 1,    # Gallup LV
    2006: 7,    # Gallup LV (RCP showed D+11; LV more accurate; actual D+7.9)
    2008: 12,   # Gallup LV
    2010: -9,   # RCP final avg R+9
    2012: 1,    # RCP avg
    2014: -2,   # RCP R+2.4
    2016: 5,    # RCP D+5
    2018: 7,    # RCP D+7.3
    2020: 7,    # RCP D+6.8
    2022: -1,   # FiveThirtyEight final avg R+0.9
    2024: 0,    # RCP R+0.3
    2026: 8.3,  # RCP avg Oct 2026
}

# Real GDP growth, Q3 year-over-year % (FRED series GDPC1).
# Positive = growing economy; negative = recession.
# Sign convention: positive always means better for the incumbent party,
# so in build_winners we flip sign when the president is Republican,
# same convention as pres_approval.
# Sources: FRED GDPC1 quarterly release; 2026 value is Q2 2026 advance estimate.
ECON_GROWTH = {
    1976:  4.9,   # Ford; strong recovery post-1975 recession
    1978:  5.3,   # Carter; expansion
    1980:  0.4,   # Carter; brief recovery after Q1-Q2 recession
    1982: -1.9,   # Reagan; deep recession (double-dip)
    1984:  6.5,   # Reagan; Morning in America boom
    1986:  3.0,   # Reagan; steady expansion
    1988:  4.4,   # Bush Sr campaign; strong
    1990:  1.6,   # Bush Sr; slowing, Gulf War
    1992:  3.8,   # Clinton campaign; recovery underway
    1994:  4.1,   # Clinton midterm; strong growth
    1996:  3.7,   # Clinton; robust expansion
    1998:  4.7,   # Clinton; dot-com boom
    2000:  3.8,   # Clinton; late expansion
    2002:  1.7,   # Bush Jr; post-recession slow recovery
    2004:  3.5,   # Bush Jr; expansion
    2006:  2.5,   # Bush Jr; solid
    2008: -0.3,   # Bush Jr; financial crisis beginning
    2010:  3.2,   # Obama; recovery (from -4% trough)
    2012:  2.5,   # Obama; steady but slow
    2014:  2.7,   # Obama; solid
    2016:  1.7,   # Obama; slowdown
    2018:  3.1,   # Trump; tax-cut boost
    2020: -2.7,   # Trump; COVID crash (Q3 rebound but still below YoY)
    2022:  1.9,   # Biden; post-COVID, pre-recession fears
    2024:  2.8,   # Biden; resilient despite rate hikes
    2026:  2.2,   # Trump; Q3 2026 YoY (FRED GDPC1)
}

# October unemployment rate (FRED series UNRATE), % of labor force.
# Sign convention: same flip as ECON_GROWTH applied in build_winners —
# high unemployment under a Rep president is Dem-favorable (positive after flip).
# Sources: FRED UNRATE monthly; 2026 value is September 2026 estimate.
UNEMPLOYMENT = {
    1976:  7.9,   # Ford; post-recession elevated
    1978:  5.9,   # Carter; improving
    1980:  7.5,   # Carter; recession surge
    1982: 10.4,   # Reagan; double-dip peak
    1984:  7.2,   # Reagan; falling but still high
    1986:  7.0,   # Reagan; gradual decline
    1988:  5.4,   # Bush Sr; near full employment
    1990:  5.9,   # Bush Sr; rising, Gulf War
    1992:  7.4,   # Bush Sr; high, election issue
    1994:  5.7,   # Clinton; rapid improvement
    1996:  5.2,   # Clinton; strong labor market
    1998:  4.5,   # Clinton; dot-com boom
    2000:  3.9,   # Clinton; historic low
    2002:  5.7,   # Bush Jr; post-recession
    2004:  5.5,   # Bush Jr; slowly improving
    2006:  4.4,   # Bush Jr; solid
    2008:  6.5,   # Bush Jr; financial crisis rising
    2010:  9.6,   # Obama; near post-war peak
    2012:  7.9,   # Obama; slow recovery
    2014:  5.8,   # Obama; improving
    2016:  4.9,   # Obama; near full employment
    2018:  3.7,   # Trump; 50-year low
    2020:  6.9,   # Trump; COVID, falling from April peak
    2022:  3.7,   # Biden; historically tight labor market
    2024:  4.1,   # Biden; slight uptick
    2026:  4.2,   # Trump; Sep 2026 (FRED UNRATE)
}

REDISTRICTING_YEARS = {1982, 1992, 2002, 2012, 2022}

# For each House election year, which presidential election result to use as pres_lean.
PRES_YEAR_FOR_HOUSE = {
    2010: 2008, 2012: 2008,
    2014: 2012, 2016: 2012,
    2018: 2016, 2020: 2016,
    2022: 2020, 2024: 2020,
}

# Incumbents not seeking re-election in 2026 (retired, running for other office, etc.)
# Source: Ballotpedia / Cook Political Report open-seat tracker, Oct 2026.
# Format: (state_po, district); at-large states use district=0.
OPEN_SEATS_2026: set[tuple[str, int]] = {
    # Democrats (26)
    ("CA", 11), ("CA", 26), ("FL", 24), ("GA", 13),
    ("IL", 2), ("IL", 4), ("IL", 7), ("IL", 8), ("IL", 9),
    ("LA", 6), ("ME", 2), ("MD", 5), ("MA", 6), ("MI", 11),
    ("MN", 2), ("NH", 1), ("NJ", 12), ("NY", 7), ("NY", 12),
    ("PA", 3), ("TN", 9), ("TX", 30), ("TX", 33), ("TX", 37),
    # Republicans (36)
    ("AL", 1), ("AZ", 1), ("AZ", 5), ("CA", 48),
    ("FL", 2), ("FL", 11), ("FL", 16), ("FL", 19),
    ("GA", 1), ("GA", 10), ("GA", 11), ("IA", 2), ("IA", 4),
    ("KY", 6), ("LA", 5), ("MI", 10), ("MO", 6), ("MT", 1),
    ("NE", 2), ("NV", 2), ("NY", 21), ("OK", 1),
    ("SC", 1), ("SC", 5), ("SD", 0), ("TN", 6),
    ("TX", 8), ("TX", 10), ("TX", 19), ("TX", 21), ("TX", 22), ("TX", 38),
    ("UT", 4), ("WA", 4), ("WI", 7), ("WY", 0),
}

if datetime.datetime.now().year > 2026:
    warnings.warn(
        "OPEN_SEATS_2026 was last updated for the 2026 cycle — update src/constants.py.",
        UserWarning,
        stacklevel=2,
    )
