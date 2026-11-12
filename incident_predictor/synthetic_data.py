"""Generates synthetic, minute-resolution service metrics with injected
"pre-incident" degradation episodes: a gradual ramp (rising CPU/latency/
error-rate) building toward a sharp incident spike, followed by recovery.

This is synthetic data for demonstrating and evaluating the prediction
pipeline end-to-end -- not a claim about any real system's behavior. The
ramp is exactly the kind of leading signal real observability stacks
(Prometheus/Grafana) surface before an outage; the point of this project is
showing a model can be trained to recognize that signal before the spike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = {
    "cpu_pct": 30.0,
    "error_rate": 0.01,
    "p95_latency_ms": 80.0,
    "request_rate": 500.0,
}

# How strongly each metric reverts toward baseline each minute (Ornstein-Uhlenbeck-style).
_REVERSION = 0.9
_NOISE_STD = {
    "cpu_pct": 1.5,
    "error_rate": 0.002,
    "p95_latency_ms": 5.0,
    "request_rate": 20.0,
}


def _simulate_baseline_walk(rng: np.random.Generator, n_minutes: int) -> dict[str, np.ndarray]:
    series = {name: np.empty(n_minutes) for name in BASELINES}
    for name, baseline in BASELINES.items():
        value = baseline
        noise_std = _NOISE_STD[name]
        values = series[name]
        for t in range(n_minutes):
            value = baseline + (value - baseline) * _REVERSION + rng.normal(0, noise_std)
            values[t] = value
    for name in series:
        if name != "request_rate":
            series[name] = np.clip(series[name], 0, None)
    return series


def _pick_incident_minutes(rng: np.random.Generator, n_minutes: int, n_incidents: int, margin: int) -> list[int]:
    usable_range = n_minutes - 2 * margin
    if n_incidents <= 0 or usable_range <= 0:
        return []
    spacing = usable_range // n_incidents
    minutes = []
    for i in range(n_incidents):
        slot_start = margin + i * spacing
        slot_end = margin + (i + 1) * spacing
        minutes.append(int(rng.integers(slot_start, slot_end)))
    return minutes


def _ease_in(fraction: float) -> float:
    """0 at fraction=0 (start of ramp), 1 at fraction=1 (incident moment); accelerates toward the end."""
    return fraction ** 2


def generate_metrics_timeseries(
    n_hours: int = 72,
    n_incidents: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_minutes = n_hours * 60

    series = _simulate_baseline_walk(rng, n_minutes)
    incident_flags = np.zeros(n_minutes, dtype=bool)

    margin = 60  # keep incidents away from the very start/end so ramp+recovery fit
    incident_minutes = _pick_incident_minutes(rng, n_minutes, n_incidents, margin)

    for t_incident in incident_minutes:
        ramp_minutes = int(rng.integers(20, 46))
        recovery_minutes = int(rng.integers(15, 31))

        ramp_start = max(0, t_incident - ramp_minutes)
        for t in range(ramp_start, t_incident):
            fraction = _ease_in((t - ramp_start) / max(1, t_incident - ramp_start))
            series["cpu_pct"][t] += fraction * 45
            series["error_rate"][t] += fraction * 0.07
            series["p95_latency_ms"][t] += fraction * 350
            series["request_rate"][t] -= fraction * 80

        series["error_rate"][t_incident] += rng.uniform(0.3, 0.5)
        series["p95_latency_ms"][t_incident] += rng.uniform(700, 1200)
        series["cpu_pct"][t_incident] = min(100.0, series["cpu_pct"][t_incident] + 40)
        incident_flags[t_incident] = True

        recovery_end = min(n_minutes, t_incident + 1 + recovery_minutes)
        for t in range(t_incident + 1, recovery_end):
            fraction = 1 - (t - t_incident) / max(1, recovery_end - t_incident)
            series["cpu_pct"][t] += fraction * 20
            series["error_rate"][t] += fraction * 0.05
            series["p95_latency_ms"][t] += fraction * 150

    df = pd.DataFrame(series)
    df["error_rate"] = df["error_rate"].clip(0, 1)
    df["request_rate"] = df["request_rate"].clip(lower=0)
    df["minute"] = np.arange(n_minutes)
    df["incident"] = incident_flags
    return df[["minute", "cpu_pct", "error_rate", "p95_latency_ms", "request_rate", "incident"]]
