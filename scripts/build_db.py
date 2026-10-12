"""One-time script: load source CSVs into data/elections.db (SQLite).

Run from project root:
    python scripts/build_db.py
"""

import sqlite3

import pandas as pd

from src.constants import Col
from src.paths import DataFile


def main() -> None:
    conn = sqlite3.connect(DataFile.DB)
    try:
        # Table: house_results
        print(f"Loading {DataFile.HOUSE} ...")
        df_house = pd.read_csv(DataFile.HOUSE, low_memory=False)
        df_house.to_sql("house_results", conn, if_exists="replace", index=False)
        print(f"  -> house_results: {len(df_house):,} rows")

        # Table: approval
        print(f"Loading {DataFile.APPROVAL} ...")
        df_approval = pd.read_csv(
            DataFile.APPROVAL,
            sep="\t",
            header=None,
            names=[Col.START, Col.END, Col.PRES_APPROVAL, Col.PRES_DISAPPROVAL, Col.UNSURE],
        )
        df_approval.to_sql("approval", conn, if_exists="replace", index=False)
        print(f"  -> approval: {len(df_approval):,} rows")

        conn.commit()
        print(f"\nDatabase written to {DataFile.DB}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
