import pandas as pd

from incident_predictor.evaluation import time_based_split


def test_time_based_split_is_positional_not_random() -> None:
    features = pd.DataFrame({"minute": range(100), "x": range(100)})
    labels = pd.Series([i % 2 == 0 for i in range(100)])

    X_train, y_train, X_test, y_test = time_based_split(features, labels, train_fraction=0.7)

    assert len(X_train) == 70
    assert len(X_test) == 30
    assert X_train["minute"].max() < X_test["minute"].min()
    assert list(X_train["minute"]) == list(range(70))
    assert list(X_test["minute"]) == list(range(70, 100))
