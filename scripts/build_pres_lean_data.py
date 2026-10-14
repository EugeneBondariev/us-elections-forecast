"""Fetch The Downballot's presidential-results-by-congressional-district data into
data/raw/president-2008-2020-by-district.csv and data/raw/president-2020-2024-by-district.csv.

ATTRIBUTION REQUIRED. Per the-downballot.com/p/data: "We're proud to make our datasets
available to the public, without cost... We welcome (almost) any use of our data as
long as you cite us and include a link to our work so that others may easily find it."
Cite The Downballot (https://www.the-downballot.com) wherever this data's results are
used or published.

The one restriction they state is "please refrain from reproducing entire spreadsheets" -
so, like scripts/build_approval_history.py, this fetches fresh from the source each run
rather than mirroring a static copy anywhere (no GitHub Release, no committing the
output - data/raw/ stays gitignored, same local-use-only treatment as every other file
built by a script in this directory).

The README's originally-documented sheet links for this data had gone stale (returned
HTTP 404); these are their current replacements, found via web search, each verified to
already match src/loaders.py::load_pres_lean's expected format exactly (same header-row
counts, same column order) - no changes needed there, just working URLs:
  - 2008-2020 (district lines as of 2020): all 4 cycles Obama/McCain through Biden/Trump.
  - 2024 (district lines as of 2024): Harris/Trump 2024 plus a 2020 comparison column.

Run from project root:
    python scripts/build_pres_lean_data.py
"""
import requests

from src.paths import DataFile

# sheet_id -> destination. Both are publicly exportable with no signup/consent gate.
SOURCES = {
    "1XbUXnI9OyfAuhP5P3vWtMuGc5UJlrhXbzZo3AwMuHtk": DataFile.PRES_08_20,
    "1ng1i_Dm_RMDnEvauH44pgE6JCUsapcuu8F2pCfeLWFo": DataFile.PRES_20_24,
}


def _fetch_sheet_csv(sheet_id: str) -> bytes:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def fetch_and_save(sheet_id: str, dest) -> int:
    """Fetch one sheet and save it byte-for-byte to dest. Returns the line count.

    write_bytes, not write_text: the response already has \\r\\n line endings, and
    write_text's default newline translation on Windows would double them to \\r\\r\\n
    (confirmed - caught by actually running this against the real download, not by a
    test written in advance). Writing raw bytes sidesteps any OS-specific translation.
    """
    content = _fetch_sheet_csv(sheet_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return content.count(b"\n")


def main() -> None:
    for sheet_id, dest in SOURCES.items():
        print(f"Fetching {dest.name} ...")
        n_lines = fetch_and_save(sheet_id, dest)
        print(f"  Wrote {dest} ({n_lines} lines)")

    print(
        "\nAttribution (required by source): data from The Downballot "
        "(https://www.the-downballot.com/p/data) - cite them wherever results derived "
        "from this are published."
    )
    print(
        "Remember: these are local-use copies fetched fresh each run, not a static "
        "mirror - never commit them or upload them anywhere public (the source asks "
        "not to reproduce entire spreadsheets)."
    )


if __name__ == "__main__":
    main()
