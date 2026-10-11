import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.forecast import run_forecast

st.set_page_config(page_title="2026 US House Forecast", layout="wide")
st.title("2026 US House Election Forecast")


@st.cache_data
def load():
    return run_forecast()


data = load()
sim = data["seats_sim"]
districts = data["districts"]
n_districts = data["n_districts"]

median_dem = int(np.median(sim))
p_majority = float((sim >= 218).mean() * 100)
ci_68_lo = int(np.percentile(sim, 16))
ci_68_hi = int(np.percentile(sim, 84))
COMPETITIVE_THRESHOLD = 0.15  # 35–65% Dem win probability
n_competitive = int(((districts["proba_dem"] - 0.5).abs() < COMPETITIVE_THRESHOLD).sum())

tab_overview, tab_districts, tab_map = st.tabs(["Overview", "Districts", "Map"])

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
        marker_color=["#3366cc" if x >= 218 else "#cc3333" for x in range(int(sim.min()), int(sim.max())+1)],
    ))
    fig.add_vline(x=217.5, line_dash="dash", line_color="black")
    fig.update_layout(showlegend=False, xaxis_title="Dem Seats", yaxis_title="Simulations",
                      margin=dict(l=20,r=20,t=20,b=20), height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Competitive districts (35–65% Dem win probability)")
    comp = (
        districts
        .assign(margin=lambda df: (df["proba_dem"] - 0.5).abs())
        .query("margin < @COMPETITIVE_THRESHOLD")
        .sort_values("margin")
        [["label", "proba_dem", "district_incumbent", "is_open"]]
        .copy()
    )
    comp["Dem win %"] = (comp["proba_dem"] * 100).round(1)
    comp["Lean"] = comp["proba_dem"].apply(lambda p: "Dem" if p >= 0.5 else "Rep")
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
    if os.getenv("ANTHROPIC_API_KEY"):
        from src.narrative import generate_narrative
        with st.spinner("Generating narrative..."):
            narrative = generate_narrative(sim, districts, n_districts)
        if narrative:
            st.write(narrative)
        else:
            st.info("Narrative generation failed — check logs.")
    else:
        st.info("Set ANTHROPIC_API_KEY to enable AI-generated narrative.")

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
                idx = df.reset_index(drop=True).index[event.selection.rows[0]]
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
    import plotly.express as px

    state_agg = (
        districts
        .groupby("state_po", as_index=False)
        .agg(
            mean_proba_dem=("proba_dem", "mean"),
            n_competitive=("proba_dem", lambda s: int(((s - 0.5).abs() < COMPETITIVE_THRESHOLD).sum())),
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
