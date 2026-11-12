"""End-to-end integration test: real synthetic data, real feature engineering,
real model training, real evaluation -- nothing mocked. This is the test that
actually validates the project's central claim: the model can flag an
incident before it happens, not just recognize it as it occurs.
"""

from incident_predictor.evaluation import evaluate, time_based_split
from incident_predictor.features import build_dataset
from incident_predictor.model import train_model
from incident_predictor.synthetic_data import generate_metrics_timeseries


def test_trained_model_detects_most_incidents_with_positive_lead_time() -> None:
    df = generate_metrics_timeseries(n_hours=120, n_incidents=16, seed=123)
    features, labels = build_dataset(df, lookback=15, lookahead=30)

    X_train, y_train, X_test, y_test = time_based_split(features, labels, train_fraction=0.7)
    feature_columns = [c for c in features.columns if c != "minute"]

    model = train_model(X_train[feature_columns], y_train)
    report = evaluate(model, X_test, y_test, df, lookahead=30)

    assert report.n_incidents_in_test >= 3, "test window should contain a meaningful number of incidents"
    assert report.n_incidents_detected >= 1, "model should flag at least one incident in advance"
    assert report.n_incidents_detected / report.n_incidents_in_test >= 0.5, (
        "model should detect at least half of in-window incidents ahead of time"
    )
    assert report.average_lead_time_minutes is not None
    assert report.average_lead_time_minutes > 0

    # Sanity bound: the model shouldn't be trivially flagging every single
    # minute as high-risk (that would inflate recall/lead-time for free).
    assert report.precision > 0.05


def test_model_outperforms_a_naive_always_predict_incident_baseline() -> None:
    """Guards against a model that's only 'detecting' incidents by firing
    constantly -- precision must beat the base incident rate in the test set."""
    df = generate_metrics_timeseries(n_hours=120, n_incidents=16, seed=123)
    features, labels = build_dataset(df, lookback=15, lookahead=30)
    X_train, y_train, X_test, y_test = time_based_split(features, labels, train_fraction=0.7)
    feature_columns = [c for c in features.columns if c != "minute"]

    model = train_model(X_train[feature_columns], y_train)
    report = evaluate(model, X_test, y_test, df, lookahead=30)

    naive_precision = y_test.mean()
    assert report.precision > naive_precision
