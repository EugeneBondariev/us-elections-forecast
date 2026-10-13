"""Parse Cook Political Report HTML ratings page → data/cook.csv."""

import csv
import re
from collections import Counter

from src.paths import DataFile

COOK_PROB = {
    "solid-d":  0.97,
    "likely-d": 0.87,
    "lean-d":   0.72,
    "tossup":   0.50,
    "toss-up":  0.50,  # handle both spellings
    "lean-r":   0.28,
    "likely-r": 0.13,
    "solid-r":  0.03,
}

DISTRICT_RE = re.compile(r'<span[^>]+class="race-district"[^>]*>\s*([^<]+?)\s*</span>')


def parse_district(raw: str):
    m = re.match(r'^([A-Z]{2})-(\d+|AL)$', raw.strip())
    if not m:
        return None
    state = m.group(1)
    district = 0 if m.group(2) == "AL" else int(m.group(2))
    return state, district


def parse(html: str) -> list[dict]:
    # Split on data-category= so each section starts with the category value
    sections = re.split(r'data-category="', html)
    rows = []
    for section in sections[1:]:
        category = section.split('"')[0]
        if category not in COOK_PROB:
            continue
        prob = COOK_PROB[category]
        for raw in DISTRICT_RE.findall(section):
            parsed = parse_district(raw)
            if parsed is None:
                continue
            state, district = parsed
            rows.append({
                "state_po": state,
                "district": district,
                "rating": category,
                "cook_prob": prob,
            })
    return rows


def main() -> None:
    html_path = DataFile.COOK.with_suffix(".html")
    csv_path = DataFile.COOK

    html = html_path.read_text(encoding="utf-8")
    rows = parse(html)

    # deduplicate — same district may appear twice if page has two grids
    seen = set()
    unique = []
    for row in rows:
        key = (row["state_po"], row["district"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state_po", "district", "rating", "cook_prob"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"Wrote {len(unique)} districts to {csv_path}")
    # quick sanity check
    counts = Counter(r["rating"] for r in unique)
    for rating in ["solid-d", "likely-d", "lean-d", "tossup", "lean-r", "likely-r", "solid-r"]:
        if counts[rating]:
            print(f"  {rating}: {counts[rating]}")


if __name__ == "__main__":
    main()
