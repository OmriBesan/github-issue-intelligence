"""Tests for classical NLP models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from issue_intelligence.models.classical import get_top_features
from issue_intelligence.models.model_registry import CLASSICAL_MODELS, get_model
from scripts.run_classical_models import load_data


def _tiny_tfidf() -> TfidfVectorizer:
    """TF-IDF with min_df=1 suitable for tiny test corpora."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,  # override for small synthetic datasets
        max_df=1.0,
        sublinear_tf=True,
    )


def _build_tiny_pipeline(model_name: str) -> Pipeline:
    """Build a pipeline with tiny-safe TF-IDF for unit tests."""
    from issue_intelligence.models.model_registry import CLASSICAL_MODELS

    clf = CLASSICAL_MODELS[model_name]().named_steps["clf"]
    return Pipeline([("tfidf", _tiny_tfidf()), ("clf", clf)])


def test_registry_contains_models() -> None:
    assert "LogisticRegression" in CLASSICAL_MODELS
    assert "LinearSVC" in CLASSICAL_MODELS
    assert "SGDClassifier" in CLASSICAL_MODELS
    assert "MultinomialNB" in CLASSICAL_MODELS


@pytest.mark.parametrize(
    "model_name",
    ["LogisticRegression", "LinearSVC", "SGDClassifier", "MultinomialNB"],
)
def test_models_fit_and_predict(model_name: str) -> None:
    # Use a larger corpus so min_df=2 doesn't strip everything
    X_train = (
        ["This is a bug error crash"] * 4
        + ["Add a new feature enhancement improvement"] * 4
        + ["Fix the docs documentation typo"] * 4
    )
    y_train = ["Bug"] * 4 + ["Enhancement"] * 4 + ["Documentation"] * 4

    pipeline = get_model(model_name)
    pipeline.fit(X_train, y_train)

    X_val = ["Another bug here", "Docs are broken"]
    preds = pipeline.predict(X_val)

    assert len(preds) == 2
    for p in preds:
        assert p in {"Bug", "Enhancement", "Documentation"}


@pytest.mark.parametrize(
    "model_name",
    ["LogisticRegression", "LinearSVC", "SGDClassifier", "MultinomialNB"],
)
def test_pipeline_steps(model_name: str) -> None:
    pipeline = get_model(model_name)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps


def test_tfidf_fitted_only_on_train() -> None:
    """Training-unique words appear in vocab; validation-only words do not."""
    X_train = [
        "trainonly trainonly trainonly commonword",
        "commonword enhancement feature add",
    ]
    y_train = ["Bug", "Enhancement"]

    pipeline = _build_tiny_pipeline("LogisticRegression")
    pipeline.fit(X_train, y_train)

    vocab = pipeline.named_steps["tfidf"].vocabulary_

    assert "trainonly" in vocab
    assert "commonword" in vocab

    # Validation-only word must NOT appear in the vocabulary
    assert "valonly" not in vocab


def test_determinism() -> None:
    X_train = ["Bug text error"] * 6 + ["Enhance text feature"] * 6
    y_train = ["Bug"] * 6 + ["Enhancement"] * 6
    X_val = ["Bug text", "Enhance text", "Unknown text"]

    model1 = get_model("LogisticRegression")
    model1.fit(X_train, y_train)
    preds1 = model1.predict(X_val)

    model2 = get_model("LogisticRegression")
    model2.fit(X_train, y_train)
    preds2 = model2.predict(X_val)

    assert (preds1 == preds2).all()


def test_linear_coefficients_extracted() -> None:
    X_train = (
        ["bug crash error traceback"] * 5
        + ["enhancement feature improve add"] * 5
        + ["documentation typo fix readme"] * 5
    )
    y_train = ["Bug"] * 5 + ["Enhancement"] * 5 + ["Documentation"] * 5

    pipeline = get_model("LinearSVC")
    pipeline.fit(X_train, y_train)

    labels_order = ["Bug", "Documentation", "Enhancement"]
    top_feats = get_top_features(pipeline, labels_order, top_n=2)

    assert "Bug" in top_feats
    assert "Documentation" in top_feats
    assert "Enhancement" in top_feats

    assert len(top_feats["Bug"]) <= 2
    assert isinstance(top_feats["Bug"][0][0], str)
    assert isinstance(top_feats["Bug"][0][1], float)


def test_multinomial_nb_no_coef_handled() -> None:
    X_train = ["bug crash"] * 5 + ["enhancement add"] * 5
    y_train = ["Bug"] * 5 + ["Enhancement"] * 5

    pipeline = _build_tiny_pipeline("MultinomialNB")
    pipeline.fit(X_train, y_train)

    labels_order = ["Bug", "Enhancement"]
    top_feats = get_top_features(pipeline, labels_order, top_n=2)
    # MultinomialNB has no coef_ attribute — should return empty dict
    assert top_feats == {}


def test_empty_data_raises_error(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()

    with pytest.raises(ValueError, match="Empty data"):
        load_data(empty_file)


def test_missing_combined_text_raises_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    with open(bad_file, "w") as f:
        f.write(json.dumps({"target": "Bug", "other_field": "text"}) + "\n")

    with pytest.raises(ValueError, match="Missing required fields"):
        load_data(bad_file)


def test_missing_target_raises_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    with open(bad_file, "w") as f:
        f.write(json.dumps({"combined_text": "text", "other_field": "text"}) + "\n")

    with pytest.raises(ValueError, match="Missing required fields"):
        load_data(bad_file)


def test_prediction_count_equals_validation_size() -> None:
    X_train = (
        ["bug error crash"] * 6
        + ["feature enhancement add"] * 6
        + ["documentation typo"] * 6
    )
    y_train = ["Bug"] * 6 + ["Enhancement"] * 6 + ["Documentation"] * 6
    X_val = ["bug bug bug", "enhancement improvement", "docs readme"]

    pipeline = get_model("LogisticRegression")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    assert len(preds) == len(X_val)
