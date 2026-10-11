import matplotlib.pyplot as plt
import numpy as np
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

tab_overview, tab_districts = st.tabs(["Overview", "Districts"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dem seats (median)", median_dem)
    c2.metric("Rep seats (median)", n_districts - median_dem)
    c3.metric("P(Dem majority)", f"{p_majority:.1f}%")
    c4.metric("Competitive (35–65% Dem)", n_competitive)

    st.subheader("Seat distribution — 10,000 simulations")
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(sim.min() - 0.5, sim.max() + 1.5, 1)
    ax.hist(sim, bins=bins, color="#1565c0", alpha=0.75)
    ax.axvspan(ci_68_lo, ci_68_hi, alpha=0.12, color="#1565c0",
               label=f"68% CI  {ci_68_lo}–{ci_68_hi}")
    ax.axvline(218, color="#333333", linestyle="--", linewidth=1.5, label="Majority (218)")
    ax.axvline(median_dem, color="#e53935", linestyle="-", linewidth=2,
               label=f"Median ({median_dem})")
    ax.set_xlabel("Democratic seats")
    ax.set_ylabel("Simulations")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

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
            prob = float(sel["proba_dem"])
            fig2, ax2 = plt.subplots(figsize=(6, 1.2))
            ax2.barh([""], [prob], color="#1565c0", height=0.5)
            ax2.barh([""], [1 - prob], left=[prob], color="#c62828", height=0.5)
            ax2.axvline(0.5, color="white", linestyle="--", linewidth=1.5)
            ax2.set_xlim(0, 1)
            ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            if prob > 0.12:
                ax2.text(prob / 2, 0, f"Dem {prob * 100:.0f}%",
                         ha="center", va="center", color="white",
                         fontweight="bold", fontsize=10)
            if (1 - prob) > 0.12:
                ax2.text(prob + (1 - prob) / 2, 0, f"Rep {(1 - prob) * 100:.0f}%",
                         ha="center", va="center", color="white",
                         fontweight="bold", fontsize=10)
            st.pyplot(fig2)
            plt.close(fig2)
