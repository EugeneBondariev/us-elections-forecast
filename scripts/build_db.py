"""One-time script: load source CSVs into data/elections.db (SQLite).

Run from project root:
    python scripts/build_db.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "elections.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Table: house_results
        house_path = DATA_DIR / "1976-2024-house.tab"
        print(f"Loading {house_path} ...")
        df_house = pd.read_csv(house_path, low_memory=False)
        df_house.to_sql("house_results", conn, if_exists="replace", index=False)
        print(f"  -> house_results: {len(df_house):,} rows")

        # Table: approval
        approval_path = DATA_DIR / "Approval-Ratings-for-POTUS-raw.xls"
        print(f"Loading {approval_path} ...")
        df_approval = pd.read_csv(
            approval_path,
            sep="\t",
            header=None,
            names=["start", "end", "pres_approval", "pres_disapproval", "unsure"],
        )
        df_approval.to_sql("approval", conn, if_exists="replace", index=False)
        print(f"  -> approval: {len(df_approval):,} rows")

        conn.commit()
        print(f"\nDatabase written to {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
