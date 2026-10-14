"""End-to-end smoke tests: start the real Streamlit app and FastAPI backend as
subprocesses, drive them with a real headless browser (Playwright), and check
for actual runtime errors. Mocked unit tests can't catch what this catches -
this suite exists because exactly this kind of check caught a real bug a mocked
test never could have: a stale/corrupted forecast cache (data/processed/elections.db
held leftover synthetic test-fixture data from an earlier manual run) that made
the Overview tab show "Rep seats: -213". Every unit test mocks run_forecast(), so
none of them touch the cache file at all.

Opt-in only - not part of the default suite (see the `e2e` marker + addopts in
pyproject.toml):
  - Needs Playwright's Chromium binary (~100MB, not part of requirements.txt):
        pip install -r requirements-dev.txt
        playwright install chromium
  - Needs real model training to complete (~60-90s cold) - much slower than the
    rest of the suite.
  - Needs all 4 required data files present (same ones conftest.py's real_data
    fixture checks for) - skips cleanly, like every other real-data test here,
    if they're missing.

Run directly:
    python -m pytest tests/test_app_e2e.py -v
or:
    make test-e2e
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from src.paths import DataFile

playwright_sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright not installed - see this file's module docstring",
)
sync_playwright = playwright_sync_api.sync_playwright

pytestmark = pytest.mark.e2e

_REPO_ROOT = Path(__file__).parent.parent
_REQUIRED_FILES = [DataFile.HOUSE, DataFile.APPROVAL, DataFile.PRES_08_20, DataFile.PRES_20_24]

# Deliberately not the real dev-server defaults (8501/8000), so this suite never
# collides with a server you already have running yourself.
STREAMLIT_PORT = 8599
API_PORT = 8598

TAB_LABELS = ["Overview", "Districts", "Map", "Ask", "Tools"]


def _wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def real_data_present():
    missing = [p.name for p in _REQUIRED_FILES if not p.exists()]
    if missing:
        pytest.skip(f"real data files not present: {missing}")


@pytest.fixture(scope="module")
def streamlit_server(real_data_present):
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            f"--server.port={STREAMLIT_PORT}", "--server.headless=true",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=_REPO_ROOT,
    )
    try:
        if not _wait_for_port(STREAMLIT_PORT, timeout=30):
            pytest.fail("Streamlit server did not start listening within 30s")
        yield f"http://127.0.0.1:{STREAMLIT_PORT}"
    finally:
        _terminate(proc)


@pytest.fixture(scope="module")
def api_server(real_data_present):
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=_REPO_ROOT,
    )
    try:
        # Longer timeout: if no cached forecast exists yet, startup itself trains
        # the full model before the port even opens.
        if not _wait_for_port(API_PORT, timeout=120):
            pytest.fail("API server did not start listening within 120s")
        yield f"http://127.0.0.1:{API_PORT}"
    finally:
        _terminate(proc)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser, streamlit_server):
    """A fresh browser CONTEXT (not just a new page) per test, already navigated
    to the app and waited for the real script run to finish.

    This matters: Streamlit ties session state (including which tab is active)
    to a cookie. Calling browser.new_page() directly shares the browser's default
    context - and its cookies - across every test, so a later test's "fresh" page
    can silently reconnect to an earlier test's session and land on whatever tab
    that one last clicked, instead of actually starting over. (Caught exactly
    this way: a seat-count assertion failed with an empty metrics list because it
    silently inherited the previous parametrized test's "Tools" tab, which has no
    metrics on it at all.) A new context per test guarantees real isolation.
    """
    context = browser.new_context()
    pg = context.new_page()
    pg.goto(streamlit_server, timeout=30000)
    pg.wait_for_selector('[data-testid="stTab"]', timeout=180000)
    time.sleep(2)  # let widgets/charts finish settling
    yield pg
    context.close()


class TestStreamlitApp:
    def test_loads_with_no_exceptions_or_console_errors(self, page):
        page_errors = []
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        exceptions = page.locator('[data-testid="stException"]').all()
        assert not exceptions, [e.inner_text() for e in exceptions]
        assert not page_errors, page_errors

    @pytest.mark.parametrize("tab_label", TAB_LABELS)
    def test_each_tab_renders_without_exceptions(self, page, tab_label):
        page.get_by_role("tab", name=tab_label).click()
        time.sleep(3)
        exceptions = page.locator('[data-testid="stException"]').all()
        assert not exceptions, f"'{tab_label}' tab: {[e.inner_text() for e in exceptions]}"

    def test_overview_shows_plausible_seat_counts(self, page):
        """Regression guard for the exact bug this suite exists because of: a
        corrupted/stale cached forecast silently produced a negative Rep-seat
        count (n_districts - median_dem with a bogus tiny n_districts)."""
        metrics = page.locator('[data-testid="stMetricValue"]').all_inner_texts()
        dem_seats = int(metrics[0])
        rep_seats = int(metrics[1])

        assert rep_seats > 0, (
            f"Rep seats was {rep_seats} - likely a stale/corrupt cached forecast "
            f"(see data/processed/elections.db, safe to delete and regenerate)"
        )
        assert dem_seats > 0
        assert 400 <= dem_seats + rep_seats <= 440  # real US House is 435 seats


class TestApi:
    def test_health_endpoint(self, api_server):
        resp = requests.get(f"{api_server}/health", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_forecast_endpoint_returns_plausible_numbers(self, api_server):
        resp = requests.get(f"{api_server}/forecast", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert 0 < data["median_dem_seats"] < 435
        assert 300 <= data["n_districts"] <= 440
        assert 0.0 <= data["p_dem_majority"] <= 1.0

    def test_districts_endpoint_returns_all_districts(self, api_server):
        resp = requests.get(f"{api_server}/districts", timeout=10)
        assert resp.status_code == 200
        districts = resp.json()
        assert len(districts) > 300
        assert {"label", "proba_dem", "district_incumbent"} <= set(districts[0].keys())

    # /refresh is deliberately not exercised here: it triggers a full synchronous
    # retrain (~20s+ even warm), which would make this already-slow suite slower
    # for no extra coverage - its auth logic is unit-tested in test_api.py, and
    # its actual behavior was verified manually against this real server already.
