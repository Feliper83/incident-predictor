"""Evaluation beyond standard classification metrics: for this problem, the
number that actually matters is lead time -- how many minutes before an
incident the model raised its first alarm. Precision/recall/F1/ROC-AUC are
reported too, but a model with great F1 and zero lead time has learned to
recognize the incident itself, not predict it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class EvaluationReport:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_incidents_in_test: int
    n_incidents_detected: int
    average_lead_time_minutes: float | None


def time_based_split(
    features: pd.DataFrame, labels: pd.Series, train_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Splits by position, not randomly -- training only ever sees data that
    precedes the test window, matching how the model would actually be used
    (never trained on the future to predict the past)."""
    split_idx = int(len(features) * train_fraction)
    return (
        features.iloc[:split_idx].reset_index(drop=True),
        labels.iloc[:split_idx].reset_index(drop=True),
        features.iloc[split_idx:].reset_index(drop=True),
        labels.iloc[split_idx:].reset_index(drop=True),
    )


def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_df: pd.DataFrame,
    lookahead: int,
    threshold: float = 0.5,
) -> EvaluationReport:
    feature_columns = [c for c in X_test.columns if c != "minute"]
    proba = model.predict_proba(X_test[feature_columns])[:, 1]
    preds = proba >= threshold

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")

    test_minutes = X_test["minute"].to_numpy()
    pred_by_minute = dict(zip(test_minutes, preds))
    test_minute_min = test_minutes.min() if len(test_minutes) else 0

    incident_minutes_in_test = raw_df.loc[
        raw_df["incident"] & raw_df["minute"].isin(test_minutes), "minute"
    ].to_numpy()

    lead_times = []
    for incident_minute in incident_minutes_in_test:
        window_start = max(incident_minute - lookahead, test_minute_min)
        alarm_minute = next(
            (m for m in range(window_start, incident_minute) if pred_by_minute.get(m)),
            None,
        )
        if alarm_minute is not None:
            lead_times.append(incident_minute - alarm_minute)

    average_lead_time = float(np.mean(lead_times)) if lead_times else None

    return EvaluationReport(
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        n_incidents_in_test=len(incident_minutes_in_test),
        n_incidents_detected=len(lead_times),
        average_lead_time_minutes=average_lead_time,
    )
