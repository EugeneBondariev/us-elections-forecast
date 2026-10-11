from fastapi import FastAPI
from src.forecast import run_forecast
import numpy as np

app = FastAPI(title="US House Forecast 2026")

@app.get("/forecast")
def forecast():
    result = run_forecast()
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
    result = run_forecast()
    df = result["districts"]
    return df[["label", "proba_dem", "district_incumbent"]].to_dict(orient="records")

@app.get("/districts/{state}/{district_num}")
def district_detail(state: str, district_num: int):
    result = run_forecast()
    df = result["districts"]
    from src.constants import Col
    row = df[(df[Col.STATE] == state.upper()) & (df[Col.DISTRICT] == district_num)]
    if row.empty:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="District not found")
    r = row.iloc[0]
    return {
        "label": r["label"],
        "proba_dem": round(float(r["proba_dem"]), 4),
        "district_incumbent": r.get("district_incumbent", "unknown"),
    }
