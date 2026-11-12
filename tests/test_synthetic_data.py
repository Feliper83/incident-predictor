from incident_predictor.synthetic_data import generate_metrics_timeseries


def test_generates_requested_number_of_incidents() -> None:
    df = generate_metrics_timeseries(n_hours=48, n_incidents=6, seed=1)
    assert df["incident"].sum() == 6


def test_output_has_expected_columns_and_length() -> None:
    df = generate_metrics_timeseries(n_hours=10, n_incidents=2, seed=1)
    assert list(df.columns) == ["minute", "cpu_pct", "error_rate", "p95_latency_ms", "request_rate", "incident"]
    assert len(df) == 10 * 60


def test_metrics_elevated_in_minutes_before_incident() -> None:
    df = generate_metrics_timeseries(n_hours=48, n_incidents=5, seed=7)
    incident_minutes = df.index[df["incident"]].to_numpy()

    for t_incident in incident_minutes:
        window = df.iloc[max(0, t_incident - 15) : t_incident]
        far_away = df[(df["minute"] < t_incident - 200) | (df["minute"] > t_incident + 200)]
        assert window["cpu_pct"].mean() > far_away["cpu_pct"].mean()
        assert window["error_rate"].mean() > far_away["error_rate"].mean()


def test_same_seed_is_reproducible() -> None:
    df1 = generate_metrics_timeseries(n_hours=24, n_incidents=3, seed=99)
    df2 = generate_metrics_timeseries(n_hours=24, n_incidents=3, seed=99)
    assert df1.equals(df2)
