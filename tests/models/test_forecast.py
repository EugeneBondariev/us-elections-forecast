class TestRunForecast:
    """End-to-end smoke test for run_forecast(). Skipped when data files are absent."""

    def test_returns_required_keys(self, forecast_result):
        assert "seats_sim" in forecast_result
        assert "districts" in forecast_result
        assert "n_districts" in forecast_result

    def test_districts_count(self, forecast_result):
        assert forecast_result["n_districts"] > 400
        assert len(forecast_result["districts"]) == forecast_result["n_districts"]

    def test_probabilities_bounded(self, forecast_result):
        proba = forecast_result["districts"]["proba_dem"]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_seat_sim_shape_and_range(self, forecast_result):
        sim = forecast_result["seats_sim"]
        assert sim.shape == (10_000,)
        assert sim.min() >= 0
        assert sim.max() <= forecast_result["n_districts"]
