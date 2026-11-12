# incident-predictor

Predicts incidents ahead of time from leading indicators in service metrics (CPU, error rate, p95 latency, request
rate) -- the same signals a Prometheus/Grafana observability stack already exposes, fed into a model trained to
recognize the ramp-up pattern that precedes an outage, instead of waiting for a human to notice it on a dashboard.

## Why this exists

Built as a focused, honestly-scoped ML project: a real scikit-learn pipeline, a methodologically correct
time-ordered train/test split (never trained on the future to predict the past), and — critically — an evaluation
metric built for what this problem actually needs. Standard classification metrics (precision/recall/F1) don't
capture whether a model is *predicting* an incident or just *recognizing* one already in progress. This project adds
a lead-time metric: for every incident in the test window, how many minutes before it occurred did the model first
raise an alarm.

## Data: synthetic, and why

The time-series data (`synthetic_data.py`) is synthetic — a baseline random walk per metric, with injected
"pre-incident" episodes: a gradual ramp (rising CPU/latency/error-rate, dropping request rate) building toward a
sharp incident spike, then recovery. This is not a claim about any specific real system's behavior; it's a
controlled way to demonstrate and evaluate the full pipeline end-to-end with a known ground truth (we know exactly
which minutes are "before an incident" because we generated them that way). The ramp shape mirrors what real
degradation looks like in production observability data — which is exactly the signal the feature engineering below
is built to catch.

## Pipeline

1. **Synthetic metrics** (`synthetic_data.py`): minute-resolution CPU/error-rate/latency/request-rate time series
   with injected incident ramps.
2. **Feature engineering** (`features.py`): for each metric, a rolling mean and a rolling **slope** (linear-fit
   trend over the trailing window) — the slope is what actually distinguishes "ramping up toward trouble" from "a
   single noisy spike that didn't lead anywhere." Labels mark every minute within `--lookahead` minutes *before* an
   incident as positive; the incident's own minute is excluded from the label, because by then there's nothing left
   to predict.
3. **Time-based split** (`evaluation.py`): the dataset is split by position (first 70% = train, last 30% = test),
   never randomly — a random split would let the model train on data chronologically *after* some of its test
   points, which no real deployment could ever do.
4. **Model** (`model.py`): a scaled, class-weighted `RandomForestClassifier`. Deliberately simple — the point of
   this project is the feature engineering and evaluation methodology, not the choice of classifier.
5. **Evaluation** (`evaluation.py`): precision/recall/F1/ROC-AUC, plus the lead-time metric described above.

## Usage

```bash
pip install -e ".[dev]"
incident-predictor run
```

Example output:

```
Simulated 4320 minutes (72h), 10 incidents. 4306 labeled rows after feature warm-up -> 3014 train / 1292 test.

Test set: precision=0.87 recall=0.64 f1=0.74 roc_auc=0.84
Incidents in test window: 3, detected in advance: 3
Average lead time before incident: 19.3 minutes
```

(Actual output from running the default command above -- not a hypothetical.)

Tune the simulation and evaluation with `--hours`, `--incidents`, `--lookback`, `--lookahead`, `--train-fraction`,
`--threshold`, `--seed`.

## What this does NOT claim

This is a demonstration of the methodology (leading-indicator features, correct time-ordered evaluation, a
lead-time metric) on synthetic data with a known, controlled signal. It is not a validated production incident
predictor — deploying something like this against real metrics would need: real incident history for labels,
leakage audits specific to your metrics pipeline, and probably a lot more negative examples of "looked scary but
wasn't" to get precision to a usable level.

## Project layout

```
incident_predictor/
  synthetic_data.py   minute-resolution metrics + injected incident ramps
  features.py          rolling mean/slope features, lookahead labels
  model.py              scaled, class-weighted RandomForestClassifier
  evaluation.py         time-based split + precision/recall/F1/ROC-AUC + lead-time metric
  cli.py                argparse entrypoint: generate -> train -> evaluate -> report
tests/
  test_synthetic_data.py         incident injection produces the intended signal
  test_features.py                label/feature correctness, including edge cases
  test_evaluation.py              split is positional, not random
  test_pipeline_integration.py    full pipeline, real model, asserts positive lead time
```

## Run the tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
