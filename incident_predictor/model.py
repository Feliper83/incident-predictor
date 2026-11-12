"""The model is intentionally simple: a scaled, class-weighted random forest.
The interesting engineering is upstream (leading-indicator features, a
methodologically time-ordered split, a lead-time evaluation metric) -- not
in reaching for the fanciest classifier.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(X_train, y_train, random_state: int = 42) -> Pipeline:
    model = build_model(random_state)
    model.fit(X_train, y_train)
    return model
