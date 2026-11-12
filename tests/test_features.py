import numpy as np
import pandas as pd

from incident_predictor.features import build_dataset, build_features, build_labels


def _flat_df(n: int, incident_at: int | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "minute": np.arange(n),
            "cpu_pct": np.full(n, 30.0),
            "error_rate": np.full(n, 0.01),
            "p95_latency_ms": np.full(n, 80.0),
            "request_rate": np.full(n, 500.0),
            "incident": np.zeros(n, dtype=bool),
        }
    )
    if incident_at is not None:
        df.loc[incident_at, "incident"] = True
    return df


def test_build_labels_marks_lookahead_window_true_and_incident_minute_false() -> None:
    df = _flat_df(100, incident_at=50)
    labels = build_labels(df, lookahead=10)

    assert labels[50] == False  # noqa: E712 - incident minute itself is not "predictable" anymore
    assert labels[40:50].all()
    assert not labels[:40].any()
    assert not labels[51:].any()


def test_build_labels_clips_lookahead_window_at_series_start() -> None:
    df = _flat_df(100, incident_at=5)
    labels = build_labels(df, lookahead=10)
    assert labels[:5].all()  # window would start at -5, clipped to 0


def test_build_features_drops_nothing_but_warmup_has_nans() -> None:
    df = _flat_df(50)
    features = build_features(df, lookback=10)
    assert features["cpu_pct_mean"].iloc[:9].isna().all()
    assert features["cpu_pct_mean"].iloc[9:].notna().all()


def test_build_features_slope_is_zero_for_constant_series() -> None:
    df = _flat_df(50)
    features = build_features(df, lookback=10)
    assert np.allclose(features["cpu_pct_slope"].iloc[9:], 0.0)


def test_build_features_slope_is_positive_for_rising_series() -> None:
    df = _flat_df(50)
    df["cpu_pct"] = np.arange(50, dtype=float)
    features = build_features(df, lookback=10)
    assert (features["cpu_pct_slope"].iloc[9:] > 0).all()


def test_build_dataset_drops_warmup_rows_and_aligns_features_with_labels() -> None:
    df = _flat_df(100, incident_at=50)
    features, labels = build_dataset(df, lookback=15, lookahead=10)

    assert len(features) == len(labels) == 100 - 14  # first 14 rows lack a full rolling window
    assert features.notna().all().all()
