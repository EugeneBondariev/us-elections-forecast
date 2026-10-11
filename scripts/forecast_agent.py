"""Interactive agentic forecast assistant using Claude tool use."""

import os
import json
import numpy as np
import pandas as pd
from src.constants import DEM_MAJORITY_SEATS

TOOLS = [
    {
        "name": "get_forecast_summary",
        "description": "Get the top-level 2026 House forecast: projected seat counts, majority probability, and confidence intervals.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_competitive_districts",
        "description": "Get the most competitive districts — those closest to 50/50.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of districts to return (default 10)", "default": 10}
            },
            "required": [],
        },
    },
    {
        "name": "get_district_detail",
        "description": "Get detailed forecast for a specific congressional district.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "Two-letter state postal code, e.g. 'PA'"},
                "district": {"type": "integer", "description": "District number. Use 0 for at-large districts."},
            },
            "required": ["state", "district"],
        },
    },
    {
        "name": "get_state_summary",
        "description": "Get all congressional districts in a state with their win probabilities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "Two-letter state postal code, e.g. 'AZ'"},
            },
            "required": ["state"],
        },
    },
]

SYSTEM = """You are a nonpartisan 2026 US House election analyst.
Use the available tools to look up forecast data before answering.
Always call a tool first — never make up numbers.
Be concise: one short paragraph per answer unless asked for more detail."""


def _run_tool(name: str, inputs: dict, forecast: dict) -> str:
    sim = forecast["seats_sim"]
    districts = forecast["districts"]
    n_districts = forecast["n_districts"]

    if name == "get_forecast_summary":
        median = int(np.median(sim))
        return json.dumps({
            "median_dem_seats": median,
            "rep_seats": n_districts - median,
            "p_dem_majority_pct": round(float((sim >= DEM_MAJORITY_SEATS).mean() * 100), 1),
            "ci_68": [int(np.percentile(sim, 16)), int(np.percentile(sim, 84))],
            "ci_95": [int(np.percentile(sim, 2.5)), int(np.percentile(sim, 97.5))],
            "n_districts": n_districts,
            "n_competitive": int(((districts["proba_dem"] - 0.5).abs() < 0.15).sum()),
        })

    if name == "get_competitive_districts":
        n = inputs.get("n", 10)
        comp = (
            districts
            .assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
            .sort_values("closeness")
            .head(n)
        )
        rows = []
        for _, r in comp.iterrows():
            rows.append({
                "label": r["label"],
                "proba_dem_pct": round(r["proba_dem"] * 100, 1),
                "lean": "Dem" if r["proba_dem"] >= 0.5 else "Rep",
                "incumbent": str(r.get("district_incumbent", "unknown")).title(),
                "is_open": bool(r.get("is_open", False)),
            })
        return json.dumps(rows)

    if name == "get_district_detail":
        state = inputs["state"].upper()
        district = inputs["district"]
        from src.constants import Col
        row = districts[(districts[Col.STATE] == state) & (districts[Col.DISTRICT] == district)]
        if row.empty:
            return json.dumps({"error": f"District {state}-{district} not found"})
        r = row.iloc[0]
        result = {
            "label": r["label"],
            "proba_dem_pct": round(float(r["proba_dem"]) * 100, 1),
            "lean": "Dem" if r["proba_dem"] >= 0.5 else "Rep",
            "incumbent_party": str(r.get("district_incumbent", "unknown")).title(),
            "is_open": bool(r.get("is_open", False)),
        }
        if "cook_rating" in r.index and str(r["cook_rating"]) != "nan":
            result["cook_rating"] = r["cook_rating"]
        if "pres_lean" in r.index:
            result["pres_lean_pct"] = round(float(r["pres_lean"]) * 100, 1)
        return json.dumps(result)

    if name == "get_state_summary":
        state = inputs["state"].upper()
        from src.constants import Col
        state_df = districts[districts[Col.STATE] == state].sort_values("proba_dem", ascending=False)
        if state_df.empty:
            return json.dumps({"error": f"No districts found for state {state}"})
        rows = []
        for _, r in state_df.iterrows():
            rows.append({
                "label": r["label"],
                "proba_dem_pct": round(float(r["proba_dem"]) * 100, 1),
                "lean": "Dem" if r["proba_dem"] >= 0.5 else "Rep",
            })
        return json.dumps({"state": state, "n_districts": len(rows), "districts": rows})

    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(user_question: str, forecast: dict, api_key: str) -> str:
    """Run one agentic turn: Claude decides which tools to call, calls them, returns final answer."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _run_tool(block.name, block.input, forecast)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return "No response generated."


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    print("Loading forecast data...")
    from src.forecast import run_forecast
    forecast = run_forecast()
    print(f"Loaded {forecast['n_districts']} districts.\n")
    print("2026 House Forecast Agent — type 'quit' to exit\n" + "-" * 50)

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = run_agent(question, forecast, api_key)
        print(f"\n{answer}")


if __name__ == "__main__":
    main()
