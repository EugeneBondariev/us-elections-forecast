"""Data setup helper for a fresh clone.

All 4 required source files (see README "Getting the data") are now obtainable
automatically, via 3 different mechanisms depending on each one's licensing situation:
  - 1976-2024-house.tab: downloaded + SHA256-verified from a GitHub Release mirror on
    this repo (see DOWNLOAD_SOURCES below). The primary source (Harvard Dataverse)
    requires a one-time Guestbook form (name/email/institution) - a deliberate consent
    gate that's out of scope to automate (it exists to track usage for the data
    provider's funding reports) - but the dataset itself is confirmed CC0 1.0 (public
    domain) via Harvard Dataverse's own API, so a mirror of an already-obtained copy is
    fine to redistribute, unlike the next one.
  - Approval-Ratings-for-POTUS-raw.xls: fetched fresh every run by
    scripts/build_approval_history.py (invoked below) from presidency.ucsb.edu's
    dedicated per-president pages. This one can NEVER get the GitHub-release-mirror
    treatment house.tab got: traced to its source, the data is "adapted from the Gallup
    Poll" (a private company) and carries an explicit "Copyright (c) The American
    Presidency Project" notice. UCSB's Terms of Service (ucsb.edu/terms-of-use) prohibit
    copying/republishing/redistributing, with exactly one exception - "one copy... for
    personal non-commercial home use." A static public mirror would be redistribution,
    which that exception doesn't cover; fetching fresh each run, locally, does not
    (same basis as a manual download) - do not add this file to DOWNLOAD_SOURCES.
  - president-2008-2020-by-district.csv / president-2020-2024-by-district.csv: fetched
    fresh every run by scripts/build_pres_lean_data.py (invoked below) from The
    Downballot's public sheets. Their stated terms explicitly welcome reuse with
    attribution ("we welcome (almost) any use of our data as long as you cite us"), with
    one restriction - "please refrain from reproducing entire spreadsheets" - so, same
    reasoning as the approval file, this fetches fresh each run rather than mirroring a
    static copy. Cite The Downballot (https://www.the-downballot.com) if you publish
    anything derived from this.

So this script checks what's present, downloads+checksum-verifies anything listed in
DOWNLOAD_SOURCES, runs the dedicated fetch script for anything that has one, falls back
to manual instructions only if that fails, and automates the genuinely-automatable
optional pieces too (FEC fundraising via FEC_API_KEY, Cook ratings parsing if you've
already saved the HTML page).

Plain ASCII output only (no em dashes / arrows) - this needs to render correctly on a
bare Windows console whose default codepage isn't UTF-8.

Run from project root:
    python scripts/fetch_data.py
or:
    make data
"""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

from src.paths import DataFile  # importing this already creates data/raw etc. - see src/paths.py

# Mirrors of CC0/public-domain required files ONLY, hosted as GitHub Release assets on
# this repo so re-downloading them doesn't require redoing a manual/consent-gated step.
# Fill in url + sha256 once the release asset exists; leave as None until then - a
# missing entry here just means _check_required() falls back to the manual instructions.
#
# DataFile.APPROVAL, DataFile.PRES_08_20, and DataFile.PRES_20_24 must NEVER be added
# here - see the module docstring above for why each one specifically can't be mirrored
# this way. Verify licensing carefully before adding anything else to this dict too.
DOWNLOAD_SOURCES = {
    DataFile.HOUSE: {
        "url": "https://github.com/EugeneBondariev/us-elections-forecast/releases/download/data-v1/1976-2024-house.tab",
        "sha256": "76eec42483ee9f0fc5cf6e377442c783d1ae6eefa929729fd6561a69bab47f2a",
    },
}


def _download_and_verify(url: str, dest: Path, sha256_hex: str) -> bool:
    """Download url to dest, verify its SHA256 matches sha256_hex. Returns True on success.

    On any failure (network error, hash mismatch) the partial/wrong file is removed
    and False is returned - callers fall back to manual instructions rather than
    silently leaving a corrupt or wrong file in place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        print(f"Downloading {dest.name} from {url} ...")
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sha256_hex:
            print(
                f"CHECKSUM MISMATCH for {dest.name}: expected {sha256_hex}, got {digest}. "
                f"Discarding the download - falling back to manual instructions."
            )
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        print(f"Verified checksum and saved: {dest}")
        return True
    except Exception as e:
        print(f"Download of {dest.name} failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


REQUIRED = [
    {
        "path": DataFile.HOUSE,
        "label": "MIT Election Lab House results (1976-2024)",
        "instructions": [
            "This file normally auto-downloads from a GitHub Release mirror on this repo"
            " (see DOWNLOAD_SOURCES above) - you're seeing manual instructions instead",
            "because that download or its checksum check just failed (network issue, or",
            "the release asset changed/moved). Fall back to the primary source:",
            "1. Open https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2",
            "2. Click Download, then fill the one-time Guestbook form (name/email/institution).",
            "   This is a mandatory Harvard Dataverse access-control step - no API bypasses it.",
            f"3. Extract the ZIP and move the .tab file to: {DataFile.HOUSE}",
        ],
    },
    {
        "path": DataFile.APPROVAL,
        "label": "Presidential approval ratings",
        "auto_script": "scripts/build_approval_history.py",
        "instructions": [
            "scripts/build_approval_history.py just ran and failed to produce this file",
            "(see its output above for why - a network issue, or presidency.ucsb.edu",
            "changed its page structure again). This one can't get a static GitHub-release",
            "mirror as a fallback, unlike house.tab - see this script's module docstring for",
            "the licensing reason - so the only option is to get the fetch script working",
            f"again. It's supposed to write directly to {DataFile.APPROVAL}.",
        ],
    },
    {
        "path": DataFile.PRES_08_20,
        "label": "Presidential results by district, 2008-2020",
        "auto_script": "scripts/build_pres_lean_data.py",
        "instructions": [
            "scripts/build_pres_lean_data.py just ran and failed to produce this file (see",
            "its output above for why - a network issue, or the source sheet moved again).",
            "Search \"The Downballot presidential results by congressional district",
            "2008-2020\" to find a current link if the one hardcoded in that script is dead,",
            f"then update it there. Expected to land at: {DataFile.PRES_08_20}",
        ],
    },
    {
        "path": DataFile.PRES_20_24,
        "label": "Presidential results by district, 2020-2024",
        "auto_script": "scripts/build_pres_lean_data.py",
        "instructions": [
            "scripts/build_pres_lean_data.py just ran and failed to produce this file (see",
            "its output above for why - a network issue, or the source sheet moved again).",
            "Search \"The Downballot presidential results by congressional district",
            "2020-2024\" to find a current link if the one hardcoded in that script is dead,",
            f"then update it there. Expected to land at: {DataFile.PRES_20_24}",
        ],
    },
]


def _check_required() -> bool:
    print("Required files (needed for `make test` to stop skipping and for the pipeline to run):\n")
    all_present = True
    ran_scripts: set[str] = set()
    for item in REQUIRED:
        present = item["path"].exists()

        if not present:
            source = DOWNLOAD_SOURCES.get(item["path"])
            if source and source["url"] and source["sha256"]:
                present = _download_and_verify(source["url"], item["path"], source["sha256"])
            elif item.get("auto_script") and item["auto_script"] not in ran_scripts:
                ran_scripts.add(item["auto_script"])
                print(f"Running {item['auto_script']} ...")
                subprocess.run([sys.executable, item["auto_script"]], check=False)
                present = item["path"].exists()
            elif item.get("auto_script"):
                # Another REQUIRED entry already ran this same script this invocation
                # (build_pres_lean_data.py fetches both Downballot files in one go).
                present = item["path"].exists()

        all_present = all_present and present
        print(f"[{'OK' if present else 'MISSING'}] {item['path'].name} - {item['label']}")
        if not present:
            for line in item["instructions"]:
                print(f"       {line}")
            print()
    return all_present


def _fetch_fundraising() -> None:
    import os

    if DataFile.FUNDRAISING.exists():
        print(f"[OK] {DataFile.FUNDRAISING.name} already present")
        return
    api_key = os.getenv("FEC_API_KEY") or os.getenv("DATA_GOV_API_KEY")
    if not api_key:
        print(
            f"[SKIP] {DataFile.FUNDRAISING.name} - optional. Set FEC_API_KEY to fetch "
            f"automatically (free key: https://api.data.gov/signup)."
        )
        return
    print("Fetching fundraising data (FEC_API_KEY is set)...")
    from src.fundraising import fetch_and_cache

    fetch_and_cache(DataFile.FUNDRAISING, api_key)
    print(f"[OK] {DataFile.FUNDRAISING.name} fetched")


def _check_cook() -> None:
    if DataFile.COOK.exists():
        print(f"[OK] {DataFile.COOK.name} already present")
        return
    html_path = DataFile.COOK_HTML
    if html_path.exists():
        print(f"Found {html_path.name} - parsing...")
        subprocess.run([sys.executable, "scripts/parse_cook.py"], check=True)
    else:
        print(
            f"[SKIP] {DataFile.COOK.name} - optional. Save the Cook Political Report House "
            f"ratings page as {html_path} (File -> Save Page As in your browser), then rerun."
        )


def main() -> None:
    print("=" * 72)
    all_required_present = _check_required()
    print("=" * 72)
    print("\nOptional data (automated where a free API key or cached file is available):\n")
    _fetch_fundraising()
    _check_cook()
    print()
    if all_required_present:
        print("All required files present - run `make test` or `python scripts/predict.py`.")
    else:
        print(
            "Some required files are still missing (see above). Each has an automated fetch\n"
            "attempt built in; if one still failed, see its specific instructions above for\n"
            "the manual fallback (this script's module docstring has the full picture)."
        )
    sys.exit(0 if all_required_present else 1)


if __name__ == "__main__":
    main()
