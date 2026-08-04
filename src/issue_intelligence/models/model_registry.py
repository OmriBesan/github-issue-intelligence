"""Model registry for easy access."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from issue_intelligence.models.classical import (
    build_linear_svc,
    build_logistic_regression,
    build_multinomial_nb,
    build_sgd_classifier,
)

CLASSICAL_MODELS = {
    "LogisticRegression": build_logistic_regression,
    "LinearSVC": build_linear_svc,
    "SGDClassifier": build_sgd_classifier,
    "MultinomialNB": build_multinomial_nb,
}

def get_model(name: str) -> Pipeline:
    """Get a fresh instance of a classical model pipeline."""
    if name in CLASSICAL_MODELS:
        return CLASSICAL_MODELS[name]()
    raise ValueError(f"Model {name} not found.")
