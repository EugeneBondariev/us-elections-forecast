from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import numpy as np
from src.forecast import run_forecast
from src.constants import Col, DEM_MAJORITY_SEATS

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.forecast = run_forecast()
    yield

app = FastAPI(title="US House Forecast 2026", lifespan=lifespan)

@app.get("/health")
def health(request: Request):
    return {"status": "ok", "cache_ready": hasattr(request.app.state, "forecast")}

@app.get("/refresh")
def refresh(request: Request):
    request.app.state.forecast = run_forecast()
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
    from fastapi import HTTPException
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
