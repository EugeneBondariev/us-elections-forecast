from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from src.forecast import run_forecast
from src.constants import Col

_cache: dict | None = None

def _get_forecast() -> dict:
    global _cache
    if _cache is None:
        _cache = run_forecast()
    return _cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_forecast()  # warm the cache on startup
    yield

app = FastAPI(title="US House Forecast 2026", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "cache_ready": _cache is not None}

@app.get("/refresh")
def refresh():
    global _cache
    _cache = None
    _get_forecast()
    return {"status": "refreshed"}

@app.get("/forecast")
def forecast():
    result = _get_forecast()
    sim = result["seats_sim"]
    return {
        "median_dem_seats": int(np.median(sim)),
        "p_dem_majority": round(float((sim >= 218).mean()), 4),
        "ci_68": [int(np.percentile(sim, 16)), int(np.percentile(sim, 84))],
        "ci_95": [int(np.percentile(sim, 2.5)), int(np.percentile(sim, 97.5))],
        "n_districts": result["n_districts"],
    }

@app.get("/districts")
def districts():
    result = _get_forecast()
    df = result["districts"]
    return df[["label", "proba_dem", "district_incumbent"]].to_dict(orient="records")

@app.get("/districts/{state}/{district_num}")
def district_detail(state: str, district_num: int):
    from fastapi import HTTPException
    result = _get_forecast()
    df = result["districts"]
    row = df[(df[Col.STATE] == state.upper()) & (df[Col.DISTRICT] == district_num)]
    if row.empty:
        raise HTTPException(status_code=404, detail="District not found")
    r = row.iloc[0]
    return {
        "label": r["label"],
        "proba_dem": round(float(r["proba_dem"]), 4),
        "district_incumbent": r.get("district_incumbent", "unknown"),
    }
