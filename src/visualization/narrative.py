"""LLM-powered forecast narrative using carefully engineered prompts."""

import logging
import os

import numpy as np
import pandas as pd

from src.constants import DEM_MAJORITY_SEATS

logger = logging.getLogger(__name__)

__all__ = ["generate_narrative"]

# Override with GROQ_MODEL if Groq retires/renames this model.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a nonpartisan political analyst writing a concise forecast briefing.
Your audience: politically-informed readers who want data-driven analysis, not spin.
Rules:
- Lead with the headline number (median Dem seats and majority probability)
- Cite specific competitive districts by name (use the provided list)
- Note structural factors: open seats, incumbency, economic environment
- Flag uncertainty honestly — mention the 95% CI
- 3 paragraphs max, no bullet points, no headers
- Neutral, analytical tone throughout"""


def _build_prompt(sim: np.ndarray, districts: pd.DataFrame, n_districts: int) -> str:
    median = int(np.median(sim))
    p_maj = float((sim >= DEM_MAJORITY_SEATS).mean() * 100)
    ci_lo = int(np.percentile(sim, 2.5))
    ci_hi = int(np.percentile(sim, 97.5))
    ci68_lo = int(np.percentile(sim, 16))
    ci68_hi = int(np.percentile(sim, 84))

    # Top competitive districts
    comp = (
        districts.assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
        .sort_values("closeness")
        .head(10)
    )
    comp_lines = []
    for r in comp.itertuples(index=False):
        inc = getattr(r, "district_incumbent", "unknown")
        open_seat = getattr(r, "is_open", False)
        tag = " (open seat)" if open_seat else f" (incumbent: {str(inc).title()})"
        comp_lines.append(f"  {r.label}: {r.proba_dem*100:.1f}% Dem{tag}")

    # Open seats count
    n_open = (
        int(districts.get("is_open", pd.Series(dtype=bool)).sum())
        if "is_open" in districts.columns
        else "unknown"
    )

    # Econ context (from dem_lean direction)
    prompt = f"""Generate a forecast briefing for the 2026 US House elections based on this data:

HEADLINE NUMBERS:
- Total contested districts: {n_districts}
- Median Democratic seats: {median} (Republicans: {n_districts - median})
- P(Democratic majority, ≥218 seats): {p_maj:.1f}%
- 68% confidence interval: {ci68_lo}–{ci68_hi} Dem seats
- 95% confidence interval: {ci_lo}–{ci_hi} Dem seats

OPEN SEATS: {n_open} districts with no incumbent running

TOP 10 MOST COMPETITIVE DISTRICTS (closest to 50/50):
{chr(10).join(comp_lines)}

Write the briefing now."""
    return prompt


def generate_narrative(sim: np.ndarray, districts: pd.DataFrame, n_districts: int) -> str:
    """Call Groq to generate a forecast narrative. Returns empty string on failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping narrative generation")
        return ""

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = _build_prompt(sim, districts, n_districts)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content.strip()
        logger.info("Narrative generated (%d chars)", len(text))
        return text
    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return ""
