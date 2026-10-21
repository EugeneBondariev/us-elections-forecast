"""Interactive agentic forecast assistant using Groq tool use."""

import json
import os
from typing import Any

import numpy as np

from src.constants import DEM_MAJORITY_SEATS, Col
from src.models.forecast import run_forecast
from src.types import ForecastResult

# Override with GROQ_MODEL if Groq retires/renames this model.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Groq's chat completions API is OpenAI-compatible: each tool is wrapped under
# {"type": "function", "function": {...}} with a "parameters" JSON schema.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast_summary",
            "description": "Get the top-level 2026 House forecast: projected seat counts, majority probability, and confidence intervals.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_competitive_districts",
            "description": "Get the most competitive districts — those closest to 50/50.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of districts to return (default 10)",
                        "default": 10,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_district_detail",
            "description": "Get detailed forecast for a specific congressional district.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Two-letter state postal code, e.g. 'PA'",
                    },
                    "district": {
                        "type": "integer",
                        "description": "District number. Use 0 for at-large districts.",
                    },
                },
                "required": ["state", "district"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state_summary",
            "description": "Get all congressional districts in a state with their win probabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Two-letter state postal code, e.g. 'AZ'",
                    },
                },
                "required": ["state"],
            },
        },
    },
]

SYSTEM = """You are a nonpartisan 2026 US House election analyst.
Use the available tools to look up forecast data before answering.
Always call a tool first — never make up numbers.
Be concise: one short paragraph per answer unless asked for more detail."""


def _run_tool(name: str, inputs: dict, forecast: ForecastResult) -> str:
    sim = forecast["seats_sim"]
    districts = forecast["districts"]
    n_districts = forecast["n_districts"]

    if name == "get_forecast_summary":
        median = int(np.median(sim))
        return json.dumps(
            {
                "median_dem_seats": median,
                "rep_seats": n_districts - median,
                "p_dem_majority_pct": round(float((sim >= DEM_MAJORITY_SEATS).mean() * 100), 1),
                "ci_68": [int(np.percentile(sim, 16)), int(np.percentile(sim, 84))],
                "ci_95": [int(np.percentile(sim, 2.5)), int(np.percentile(sim, 97.5))],
                "n_districts": n_districts,
                "n_competitive": int(((districts["proba_dem"] - 0.5).abs() < 0.15).sum()),
            }
        )

    if name == "get_competitive_districts":
        n = inputs.get("n", 10)
        comp = (
            districts.assign(closeness=lambda df: (df["proba_dem"] - 0.5).abs())
            .sort_values("closeness")
            .head(n)
        )
        rows = []
        for r in comp.to_dict("records"):
            proba = float(r["proba_dem"])
            rows.append(
                {
                    "label": r["label"],
                    "proba_dem_pct": round(proba * 100, 1),
                    "lean": "Dem" if proba >= 0.5 else "Rep",
                    "incumbent": str(r.get("district_incumbent", "unknown")).title(),
                    "is_open": bool(r.get("is_open", False)),
                }
            )
        return json.dumps(rows)

    if name == "get_district_detail":
        state = inputs["state"].upper()
        district = inputs["district"]
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
        state_df = districts[districts[Col.STATE] == state].sort_values(
            "proba_dem", ascending=False
        )
        if state_df.empty:
            return json.dumps({"error": f"No districts found for state {state}"})
        rows = []
        for r in state_df.to_dict("records"):
            proba = float(r["proba_dem"])
            rows.append(
                {
                    "label": r["label"],
                    "proba_dem_pct": round(proba * 100, 1),
                    "lean": "Dem" if proba >= 0.5 else "Rep",
                }
            )
        return json.dumps({"state": state, "n_districts": len(rows), "districts": rows})

    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(user_question: str, forecast: ForecastResult, api_key: str) -> str:
    """Run one agentic turn: the model decides which tools to call, calls them, returns the final answer."""
    from groq import Groq

    client = Groq(api_key=api_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_question},
    ]

    while True:
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            max_tokens=1024,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "No response generated."

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )
        for tc in message.tool_calls:
            inputs = json.loads(tc.function.arguments)
            result = _run_tool(tc.function.name, inputs, forecast)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        return

    print("Loading forecast data...")
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
