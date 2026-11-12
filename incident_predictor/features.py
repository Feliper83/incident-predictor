"""Turns raw per-minute metrics into a supervised learning dataset: rolling
leading-indicator features, and a label for "does an incident happen within
the next `lookahead` minutes" -- using only information available up to and
including the current minute, so the model is genuinely predicting the
future rather than peeking at it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = ["cpu_pct", "error_rate", "p95_latency_ms", "request_rate"]


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of a simple linear fit over the trailing `window` points -- the
    leading indicator that actually distinguishes a ramp-up from a one-off
    noisy spike."""
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(values: np.ndarray) -> float:
        return float(((x - x_mean) * (values - values.mean())).sum() / denom)

    return series.rolling(window).apply(slope, raw=True)


def build_features(df: pd.DataFrame, lookback: int = 15) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["minute"] = df["minute"]
    for column in METRIC_COLUMNS:
        features[f"{column}_current"] = df[column]
        features[f"{column}_mean"] = df[column].rolling(lookback).mean()
        features[f"{column}_slope"] = _rolling_slope(df[column], lookback)
    return features


def build_labels(df: pd.DataFrame, lookahead: int = 30) -> pd.Series:
    """label[t] = True iff an incident occurs at some minute in (t, t+lookahead].
    The incident's own minute is labeled False -- by then it has already
    happened, so it isn't something to *predict* anymore."""
    incident = df["incident"].to_numpy()
    n = len(incident)
    label = np.zeros(n, dtype=bool)
    for idx in np.flatnonzero(incident):
        start = max(0, idx - lookahead)
        label[start:idx] = True
    return pd.Series(label, index=df.index, name="label")


def build_dataset(df: pd.DataFrame, lookback: int = 15, lookahead: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (features, labels), dropping the warm-up rows where the
    rolling window isn't full yet (their features would be NaN)."""
    features = build_features(df, lookback=lookback)
    labels = build_labels(df, lookahead=lookahead)
    valid = features.notna().all(axis=1)
    return features[valid].reset_index(drop=True), labels[valid].reset_index(drop=True)
