"""Baseline models for issue classification."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier


class MajorityClassBaseline:
    """Always predicts the most frequent class in the training data."""

    def __init__(self) -> None:
        self.model = DummyClassifier(strategy="most_frequent")
        self.is_fitted = False
        self.classes_: list[str] = []

    def fit(self, y: list[str]) -> None:
        """Fit the baseline model on training labels."""
        if not y:
            raise ValueError("Training labels cannot be empty")
            
        # DummyClassifier requires an X array, but most_frequent strategy ignores its contents.
        # Provide a dummy 2D array of the same length as y.
        X = np.zeros((len(y), 1))
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_.tolist()

    def predict(self, n_samples: int) -> list[str]:
        """Predict the majority class for n_samples.
        
        We accept `n_samples` rather than a feature array since the baseline
        doesn't use features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
            
        if n_samples <= 0:
            return []
            
        X = np.zeros((n_samples, 1))
        return self.model.predict(X).tolist()


class StratifiedRandomBaseline:
    """Predicts classes randomly according to their training distribution."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = DummyClassifier(strategy="stratified", random_state=random_state)
        self.is_fitted = False
        self.classes_: list[str] = []

    def fit(self, y: list[str]) -> None:
        """Fit the baseline model to calculate class probabilities."""
        if not y:
            raise ValueError("Training labels cannot be empty")
            
        X = np.zeros((len(y), 1))
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_.tolist()

    def predict(self, n_samples: int) -> list[str]:
        """Predict classes based on training distribution."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
            
        if n_samples <= 0:
            return []
            
        X = np.zeros((n_samples, 1))
        return self.model.predict(X).tolist()
