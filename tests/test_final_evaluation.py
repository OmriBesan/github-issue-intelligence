"""Tests for the final test set evaluation script (Stage 4B)."""

import json
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest

from scripts.evaluate_final_model import LABELS, build_final_pipeline, main


@pytest.fixture
def synthetic_splits(tmp_path):
    """Create synthetic JSONL files for testing."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    train_data = [
        {"issue_id": 1, "combined_text": "fix a bug", "target": "Bug"},
        {"issue_id": 2, "combined_text": "update docs", "target": "Documentation"},
    ]
    val_data = [
        {"issue_id": 3, "combined_text": "add feature", "target": "Enhancement"},
    ]
    # The test script strictly requires 856 records for the test set.
    # We create exactly 856 synthetic records to pass the length check.
    test_data = [
        {"issue_id": 100 + i, "combined_text": f"test bug {i}", "target": "Bug"}
        for i in range(856)
    ]

    temporal_dir = splits_dir / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_data).to_json(
        temporal_dir / "train.jsonl", orient="records", lines=True
    )
    pd.DataFrame(val_data).to_json(
        temporal_dir / "validation.jsonl", orient="records", lines=True
    )
    pd.DataFrame(test_data).to_json(
        temporal_dir / "test.jsonl", orient="records", lines=True
    )

    return splits_dir


def test_locked_model_configuration():
    """Verify the exact locked model configuration is used."""
    pipeline = build_final_pipeline()

    # 1. Pipeline structure
    assert list(pipeline.named_steps.keys()) == ["tfidf", "clf"]

    # 2. TF-IDF config
    tfidf = pipeline.named_steps["tfidf"]
    assert tfidf.lowercase is True
    assert tfidf.ngram_range == (1, 2)
    assert tfidf.min_df == 5
    assert tfidf.max_df == 0.95
    assert tfidf.sublinear_tf is True
    assert tfidf.max_features is None

    # 3. Classifier config
    clf = pipeline.named_steps["clf"]
    assert clf.C == 0.3
    assert clf.class_weight == "balanced"


def test_label_order_is_fixed():
    """Verify that output labels are strictly ordered."""
    assert LABELS == ["Bug", "Documentation", "Enhancement"]


def test_main_eval_flow(synthetic_splits, tmp_path):
    """Test the complete evaluation flow with synthetic data."""
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"

    # Mock argparse to use our temp dirs
    test_args = MagicMock()
    test_args.splits_dir = synthetic_splits
    test_args.models_dir = models_dir
    test_args.results_dir = results_dir
    test_args.figures_dir = figures_dir

    with patch("argparse.ArgumentParser.parse_args", return_value=test_args):
        # We need to temporarily patch tfidf min_df since our synthetic set is tiny
        with patch("scripts.evaluate_final_model.build_final_pipeline") as mock_build:
            # Create pipeline with min_df=1 for synthetic test
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline
            from sklearn.svm import LinearSVC

            p = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(min_df=1)),
                    ("clf", LinearSVC(C=0.3, class_weight="balanced")),
                ]
            )
            mock_build.return_value = p
            main()

    # Assert outputs created
    assert (models_dir / "final_linear_svc.joblib").exists()
    assert (results_dir / "final_temporal_test.json").exists()
    assert (results_dir / "final_model_summary.csv").exists()
    assert (figures_dir / "final_test_confusion_matrix.png").exists()
    assert (figures_dir / "final_validation_vs_test.png").exists()

    # Verify JSON schema
    with open(results_dir / "final_temporal_test.json") as f:
        data = json.load(f)

    assert data["model_name"] == "LinearSVC"
    assert "macro_f1" in data["test_metrics"]
    assert set(data["test_metrics"]["per_class"].keys()) == set(LABELS)
    assert data["feature_matrix_shape"][0] == 3  # 2 train + 1 val
    assert sum(data["predicted_distribution"].values()) == 856


def test_overlapping_ids_rejected(tmp_path):
    """Test that train/test overlapping issue IDs raise an error."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    temporal_dir = splits_dir / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)

    # Both train and test have issue_id=1
    pd.DataFrame(
        [{"issue_id": 1, "combined_text": "txt", "target": "Bug"}]
    ).to_json(temporal_dir / "train.jsonl", orient="records", lines=True)
    pd.DataFrame(
        [{"issue_id": 2, "combined_text": "txt", "target": "Bug"}]
    ).to_json(temporal_dir / "validation.jsonl", orient="records", lines=True)
    test_data = [
        {"issue_id": 1 if i == 0 else 100 + i, "combined_text": "t", "target": "Bug"}
        for i in range(856)
    ]
    pd.DataFrame(test_data).to_json(
        temporal_dir / "test.jsonl", orient="records", lines=True
    )

    test_args = MagicMock()
    test_args.splits_dir = splits_dir

    with patch("argparse.ArgumentParser.parse_args", return_value=test_args):
        with pytest.raises(ValueError, match="overlapping between train and test"):
            main()


def test_duplicate_train_ids_rejected(tmp_path):
    """Test that duplicate issue IDs in train+val raise an error."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    temporal_dir = splits_dir / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)

    # Both train and val have issue_id=1
    pd.DataFrame([{"issue_id": 1, "combined_text": "a", "target": "Bug"}]).to_json(
        temporal_dir / "train.jsonl", orient="records", lines=True
    )
    pd.DataFrame([{"issue_id": 1, "combined_text": "b", "target": "Bug"}]).to_json(
        temporal_dir / "validation.jsonl", orient="records", lines=True
    )
    test_data = [
        {"issue_id": 100 + i, "combined_text": "t", "target": "Bug"} for i in range(856)
    ]
    pd.DataFrame(test_data).to_json(
        temporal_dir / "test.jsonl", orient="records", lines=True
    )

    test_args = MagicMock()
    test_args.splits_dir = splits_dir

    with patch("argparse.ArgumentParser.parse_args", return_value=test_args):
        with pytest.raises(ValueError, match="duplicate issue_ids"):
            main()


def test_deterministic_predictions():
    """Verify that predictions with a fixed random_state are deterministic."""
    p1 = build_final_pipeline()
    p2 = build_final_pipeline()

    # Temporarily set min_df=1 for small inputs
    p1.named_steps["tfidf"].min_df = 1
    p2.named_steps["tfidf"].min_df = 1

    X = ["bug crash", "update docs", "add new feature"]
    y = ["Bug", "Documentation", "Enhancement"]

    p1.fit(X, y)
    p2.fit(X, y)

    X_test = ["crash", "docs", "new"]
    assert (p1.predict(X_test) == p2.predict(X_test)).all()


def test_empty_input_handling():
    """Verify the model handles empty string inputs."""
    pipeline = build_final_pipeline()
    pipeline.named_steps["tfidf"].min_df = 1

    pipeline.fit(["crash", "docs", "feature"], ["Bug", "Documentation", "Enhancement"])

    # Should not crash on empty input, should return a valid string label
    pred = pipeline.predict([""])
    assert len(pred) == 1
    assert pred[0] in LABELS


def test_model_save_load_behavior(tmp_path):
    """Verify that the joblib saved model can be reloaded and predicts identically."""
    pipeline = build_final_pipeline()
    pipeline.named_steps["tfidf"].min_df = 1
    pipeline.fit(["crash", "docs", "feature"], ["Bug", "Documentation", "Enhancement"])

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    loaded_pipeline = joblib.load(model_path)
    X_test = ["crash", "feature", "docs"]
    assert (pipeline.predict(X_test) == loaded_pipeline.predict(X_test)).all()


def test_unknown_label_handling():
    """Verify confusion matrix handles true labels outside of LABELS gracefully
    by passing labels=LABELS to confusion_matrix."""
    from sklearn.metrics import confusion_matrix
    y_true = ["Bug", "UnknownClass"]
    y_pred = ["Bug", "Bug"]

    # Passing labels=LABELS should ignore the 'UnknownClass' true label for matrix creation
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # Bug->Bug
