import os
import subprocess
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.constants import COMPETITIVE_MARGIN, DEM_MAJORITY_SEATS
from src.data.database import load_latest_forecast, save_forecast
from src.models.forecast import run_forecast
from src.paths import DataFile
from src.visualization.narrative import generate_narrative
from src.visualization.rag import build_index, load_index, rag_answer

st.set_page_config(page_title="2026 US House Forecast", layout="wide")
st.title("2026 US House Election Forecast")


_REQUIRED_COLS = {"label", "proba_dem", "district_incumbent", "is_open", "dem_lean", "state_po"}


@st.cache_data
def load():
    cached = load_latest_forecast(DataFile.DB)
    if cached is not None and _REQUIRED_COLS.issubset(cached["districts"].columns):
        return cached
    result = run_forecast()
    save_forecast(result, DataFile.DB)
    get_rag_collection.clear()
    return result


@st.cache_resource
def get_rag_collection(_districts):
    try:
        return load_index()
    except Exception:
        return build_index(_districts)


data = load()
sim = data["seats_sim"]
districts = data["districts"]
n_districts = data["n_districts"]

median_dem = int(np.median(sim))
p_majority = float((sim >= DEM_MAJORITY_SEATS).mean() * 100)
ci_68_lo = int(np.percentile(sim, 16))
ci_68_hi = int(np.percentile(sim, 84))
n_competitive = int(((districts["proba_dem"] - 0.5).abs() < COMPETITIVE_MARGIN).sum())

tab_overview, tab_districts, tab_map, tab_ask, tab_tools = st.tabs(["Overview", "Districts", "Map", "Ask", "Tools"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dem seats (median)", median_dem)
    c2.metric("Rep seats (median)", n_districts - median_dem)
    c3.metric("P(Dem majority)", f"{p_majority:.1f}%")
    c4.metric("Competitive (35–65% Dem)", n_competitive)

    st.subheader("Seat distribution — 10,000 simulations")
    fig = go.Figure(go.Histogram(
        x=sim,
        xbins=dict(start=sim.min()-0.5, end=sim.max()+0.5, size=1),
        marker_color=["#3366cc" if x >= DEM_MAJORITY_SEATS else "#cc3333" for x in range(int(sim.min()), int(sim.max())+1)],
    ))
    fig.add_vline(x=DEM_MAJORITY_SEATS - 0.5, line_dash="dash", line_color="black")
    fig.update_layout(showlegend=False, xaxis_title="Dem Seats", yaxis_title="Simulations",
                      margin=dict(l=20,r=20,t=20,b=20), height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Competitive districts (35–65% Dem win probability)")
    comp = (
        districts
        .assign(margin=lambda df: (df["proba_dem"] - 0.5).abs())
        .query("margin < @COMPETITIVE_MARGIN")
        .sort_values("margin")
        [["label", "proba_dem", "district_incumbent", "is_open"]]
        .copy()
    )
    comp["Dem win %"] = (comp["proba_dem"] * 100).round(1)
    comp["Lean"] = np.where(comp["proba_dem"] >= 0.5, "Dem", "Rep")
    comp = comp.rename(columns={
        "label": "District",
        "district_incumbent": "Incumbent party",
        "is_open": "Open seat",
    })
    st.dataframe(
        comp[["District", "Dem win %", "Lean", "Incumbent party", "Open seat"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Forecast narrative")
    if os.getenv("GROQ_API_KEY"):
        with st.spinner("Generating narrative..."):
            narrative = generate_narrative(sim, districts, n_districts)
        if narrative:
            st.write(narrative)
        else:
            st.info("Narrative generation failed — check logs.")
    else:
        st.info("Set GROQ_API_KEY to enable AI-generated narrative.")

with tab_districts:
    f1, f2, f3 = st.columns([1, 1, 1])
    state_options = ["All"] + sorted(districts["state_po"].unique().tolist())
    state_sel = f1.selectbox("State", state_options)
    party_sel = f2.selectbox("Incumbent party", ["All", "Democrat", "Republican"])
    open_sel = f3.checkbox("Open seats only")

    df = districts.copy()
    if state_sel != "All":
        df = df[df["state_po"] == state_sel]
    if party_sel == "Democrat":
        df = df[df["district_incumbent"] == "DEMOCRAT"]
    elif party_sel == "Republican":
        df = df[df["district_incumbent"] == "REPUBLICAN"]
    if open_sel:
        df = df[df["is_open"]]

    display = df[["label", "proba_dem", "district_incumbent", "is_open", "dem_lean"]].copy()
    display["proba_dem"] = (display["proba_dem"] * 100).round(1)
    display["dem_lean"] = (display["dem_lean"] * 100).round(1)
    display["district_incumbent"] = display["district_incumbent"].str.title()
    display.columns = ["District", "Dem win %", "Incumbent party", "Open seat", "Dem lean (hist %)"]
    display = display.sort_values("Dem win %", ascending=False).reset_index(drop=True)

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:
        idx = event.selection.rows[0]
        sel = df.reset_index(drop=True).iloc[idx]
        st.divider()
        dcol1, dcol2 = st.columns([1, 2])

        with dcol1:
            st.subheader(sel["label"])
            st.metric("Dem win probability", f"{sel['proba_dem'] * 100:.1f}%")
            st.write(f"**Incumbent party:** {sel['district_incumbent'].title()}")
            st.write(f"**Open seat:** {'Yes' if sel['is_open'] else 'No'}")
            st.write(f"**Historical Dem lean:** {sel['dem_lean'] * 100:.1f}%")
            if "pres_lean" in sel.index and not np.isnan(float(sel["pres_lean"])):
                st.write(f"**Presidential lean (2020/24):** {sel['pres_lean'] * 100:.1f}%")

        with dcol2:
            row = sel
            fig2 = go.Figure(go.Bar(
                x=[row["proba_dem"]],
                orientation="h",
                marker_color="#3366cc" if row["proba_dem"] >= 0.5 else "#cc3333",
            ))
            fig2.update_layout(xaxis=dict(range=[0,1], title="P(Dem win)"),
                               height=80, margin=dict(l=0,r=0,t=0,b=0),
                               showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            if "shap_values" in data:
                shap_df = data["shap_values"]
                row_pos = event.selection.rows[0]
                if row_pos < len(shap_df):
                    sv = shap_df.iloc[row_pos].sort_values(key=abs, ascending=False).head(5)
                    fig_shap = go.Figure(go.Bar(
                        x=sv.values,
                        y=sv.index,
                        orientation="h",
                        marker_color=["#3366cc" if v > 0 else "#cc3333" for v in sv.values],
                    ))
                    fig_shap.update_layout(
                        title="Top SHAP drivers",
                        height=250,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="SHAP value (+ = Dem)",
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

with tab_map:
    state_agg = (
        districts
        .groupby("state_po", as_index=False)
        .agg(
            mean_proba_dem=("proba_dem", "mean"),
            n_competitive=("proba_dem", lambda s: int(((s - 0.5).abs() < COMPETITIVE_MARGIN).sum())),
        )
    )

    fig_map = px.choropleth(
        state_agg,
        locations="state_po",
        locationmode="USA-states",
        color="mean_proba_dem",
        color_continuous_scale="RdBu",
        range_color=[0.2, 0.8],
        scope="usa",
        title="Average Dem win probability by state",
        labels={"mean_proba_dem": "Avg Dem %"},
    )
    fig_map.update_layout(height=500)
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Most competitive states")
    top_competitive = (
        state_agg
        .sort_values("n_competitive", ascending=False)
        .head(5)
        .copy()
    )
    top_competitive["Avg Dem %"] = (top_competitive["mean_proba_dem"] * 100).round(1)
    top_competitive = top_competitive.rename(columns={
        "state_po": "State",
        "n_competitive": "Competitive districts",
    })
    st.dataframe(
        top_competitive[["State", "Avg Dem %", "Competitive districts"]],
        use_container_width=False,
        hide_index=True,
    )

with tab_ask:
    st.subheader("Ask about the forecast")
    if not os.getenv("GROQ_API_KEY"):
        st.info("Set GROQ_API_KEY to enable natural language Q&A.")
    else:
        with st.expander("What can I ask?", expanded=False):
            st.markdown(
                """
The assistant answers questions based on the **current model forecast** for each district.
Each district has: win probability, incumbent party, open seat flag, historical Dem lean,
presidential lean, and Cook Political Report rating (where available).

**Good questions:**
- *Which districts in Pennsylvania are competitive?*
- *Which open seats does the model favor Republicans in?*
- *What are the closest toss-up races in Florida?*
- *Which Democratic-held districts have a win probability below 50%?*
- *How does the model rate swing districts in the Midwest?*

**Won't work well:**
- Questions about specific candidate names (not in the data)
- Historical questions about past elections
- Questions about polling, fundraising totals, or news events
- National-level questions like "who will control the House?" — use the Overview tab for that
                """
            )

        query = st.text_input(
            "Ask a question about the 2026 House forecast",
            placeholder="e.g. Which Pennsylvania districts are competitive?",
        )
        if query:
            with st.spinner("Searching..."):
                answer = rag_answer(query, collection=get_rag_collection(districts))
            st.write(answer)

with tab_tools:
    st.subheader("Pipeline tools")

    _SCRIPTS = {
        "Run full analysis":       ("src.visualization.run_analysis", "Runs all 6 diagnostic scripts. Logs saved to logs/analysis/."),
        "Train & predict (MLflow)":("src.models.predict",             "Full training run with walk-forward CV, bootstrap CI, and 2026 forecast. Logged to MLflow."),
        "Hyperparameter tuning":   ("src.models.tune",                "Optuna tuning for LightGBM (500 trials). Saves best params to config/best_params_lgbm.json."),
        "Calibrate sigma":         ("src.models.calibrate_sigma",     "Fits Monte Carlo sigma_national to walk-forward residuals. Saves to data/calibrated_sigma.json."),
        "Rebuild RAG index":       ("src.visualization.build_rag_index", "Rebuilds the ChromaDB vector index from the current forecast."),
        "Fetch economic indicators":("src.data.build_indicators",    "Downloads latest FRED GDP and unemployment data to data/interim/indicators.csv."),
        "Parse Cook ratings":      ("src.data.parse_cook",            "Parses Cook Political Report CSV into data/cook.csv."),
    }

    for label, (module, description) in _SCRIPTS.items():
        with st.expander(label):
            st.caption(description)
            if st.button(f"Run — {label}", key=label):
                with st.spinner(f"Running {module}..."):
                    result = subprocess.run(
                        ["python", "-m", module],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                if result.returncode == 0:
                    st.success(f"Finished at {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.error("Script failed — see output below")
                if result.stdout:
                    st.code(result.stdout[-3000:], language="text")
                if result.stderr:
                    st.code(result.stderr[-1000:], language="text")
