import pytest


@pytest.mark.usefixtures("real_data")
class TestRunForecast:
    """End-to-end smoke test for run_forecast(). Skipped when data files are absent."""

    def test_returns_required_keys(self):
        from src.models.forecast import run_forecast
        result = run_forecast()
        assert "seats_sim" in result
        assert "districts" in result
        assert "n_districts" in result

    def test_districts_count(self):
        from src.models.forecast import run_forecast
        result = run_forecast()
        assert result["n_districts"] > 400
        assert len(result["districts"]) == result["n_districts"]

    def test_probabilities_bounded(self):
        from src.models.forecast import run_forecast
        result = run_forecast()
        proba = result["districts"]["proba_dem"]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_seat_sim_shape_and_range(self):
        from src.models.forecast import run_forecast
        result = run_forecast()
        sim = result["seats_sim"]
        assert sim.shape == (10_000,)
        assert sim.min() >= 0
        assert sim.max() <= result["n_districts"]
