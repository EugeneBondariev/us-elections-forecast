from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from src.constants import Col, DEM_MAJORITY_SEATS
from src.database import load_latest_forecast, save_forecast
from src.forecast import run_forecast

@asynccontextmanager
async def lifespan(app: FastAPI):
    result = load_latest_forecast()
    if result is None:
        result = run_forecast()
        save_forecast(result)
    app.state.forecast = result
    yield

app = FastAPI(title="US House Forecast 2026", lifespan=lifespan)

@app.get("/health")
def health(request: Request):
    return {"status": "ok", "cache_ready": hasattr(request.app.state, "forecast")}

@app.get("/refresh")
def refresh(request: Request):
    result = run_forecast()
    save_forecast(result)
    request.app.state.forecast = result
    return {"status": "refreshed"}

@app.get("/forecast")
def forecast(request: Request):
    result = request.app.state.forecast
    sim = result["seats_sim"]
    return {
        "median_dem_seats": int(np.median(sim)),
        "p_dem_majority": round(float((sim >= DEM_MAJORITY_SEATS).mean()), 4),
        "ci_68": [int(np.percentile(sim, 16)), int(np.percentile(sim, 84))],
        "ci_95": [int(np.percentile(sim, 2.5)), int(np.percentile(sim, 97.5))],
        "n_districts": result["n_districts"],
    }

@app.get("/districts")
def districts(request: Request):
    df = request.app.state.forecast["districts"]
    return df[["label", "proba_dem", "district_incumbent"]].to_dict(orient="records")

@app.get("/districts/{state}/{district_num}")
def district_detail(state: str, district_num: int, request: Request):
    df = request.app.state.forecast["districts"]
    row = df[(df[Col.STATE] == state.upper()) & (df[Col.DISTRICT] == district_num)]
    if row.empty:
        raise HTTPException(status_code=404, detail="District not found")
    r = row.iloc[0]
    return {
        "label": r["label"],
        "proba_dem": round(float(r["proba_dem"]), 4),
        "district_incumbent": r.get("district_incumbent", "unknown"),
    }
