"""Tests for baseline models."""

from __future__ import annotations

import pytest

from issue_intelligence.models.baselines import (
    MajorityClassBaseline,
    StratifiedRandomBaseline,
)


def test_majority_class_predicts_majority() -> None:
    y_train = ["Bug", "Bug", "Enhancement"]
    model = MajorityClassBaseline()

    with pytest.raises(RuntimeError):
        model.predict(3)

    model.fit(y_train)

    # Should always predict "Bug" (the majority class)
    preds = model.predict(5)
    assert len(preds) == 5
    assert all(p == "Bug" for p in preds)


def test_stratified_baseline_is_reproducible() -> None:
    y_train = ["Bug"] * 60 + ["Documentation"] * 30 + ["Enhancement"] * 10

    model1 = StratifiedRandomBaseline(random_state=42)
    model1.fit(y_train)
    preds1 = model1.predict(100)

    model2 = StratifiedRandomBaseline(random_state=42)
    model2.fit(y_train)
    preds2 = model2.predict(100)

    assert preds1 == preds2


def test_stratified_baseline_proportions() -> None:
    y_train = ["Bug"] * 800 + ["Enhancement"] * 200

    model = StratifiedRandomBaseline(random_state=42)
    model.fit(y_train)

    # Predict a large number of samples to check distribution
    preds = model.predict(1000)

    bug_count = preds.count("Bug")
    enh_count = preds.count("Enhancement")

    # Should be roughly 80% Bug, 20% Enhancement
    # Note: dummy classifier is probabilistic, so we check an approximate range
    assert 750 <= bug_count <= 850
    assert 150 <= enh_count <= 250


def test_empty_train_raises_error() -> None:
    model = MajorityClassBaseline()
    with pytest.raises(ValueError, match="Training labels cannot be empty"):
        model.fit([])


def test_zero_samples_prediction() -> None:
    y_train = ["Bug"] * 3
    model = MajorityClassBaseline()
    model.fit(y_train)

    assert model.predict(0) == []
    assert model.predict(-5) == []
