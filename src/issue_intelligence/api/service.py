"""Inference service logic."""

import logging
from pathlib import Path
from typing import Any

import joblib

from issue_intelligence.data.preprocessing import (
    build_combined_text,
    clean_body,
    clean_title,
)

logger = logging.getLogger(__name__)


class InferenceService:
    """Service to load and evaluate the text classifier."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline: Any = None
        self.is_ready = False
        self.classes_: list[str] = []
        self.vocab_size: int = 0
        self.model_name = "LinearSVC"

    def load_model(self) -> None:
        """Load the model pipeline from disk."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            self.is_ready = False
            return

        try:
            self.pipeline = joblib.load(self.model_path)

            # Extract classes directly from the fitted classifier
            clf = self.pipeline.named_steps.get("clf")
            if clf is not None and hasattr(clf, "classes_"):
                self.classes_ = clf.classes_.tolist()
            else:
                self.classes_ = ["Bug", "Documentation", "Enhancement"]

            # Try to get vocab size if TF-IDF is present
            tfidf = self.pipeline.named_steps.get("tfidf")
            if tfidf is not None and hasattr(tfidf, "vocabulary_"):
                self.vocab_size = len(tfidf.vocabulary_)

            self.is_ready = True
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load corrupt model from {self.model_path}: {e}")
            self.is_ready = False
            self.pipeline = None

    def predict(self, title: str | None, body: str | None) -> dict[str, Any]:
        """Run prediction and calculate raw decision scores and margins."""
        if not self.is_ready or self.pipeline is None:
            raise RuntimeError("Model is not loaded.")

        clean_t, _ = clean_title(title)
        clean_b = clean_body(body)
        combined = build_combined_text(clean_t, clean_b)

        # We must have non-empty text (should be handled by validator, but double check)
        if not combined:
            raise ValueError("Combined text is empty after preprocessing.")

        # Raw scores from decision_function
        # For a 3-class model, shape is (1, 3)
        X = [combined]

        # Predict the top class
        y_pred = self.pipeline.predict(X)[0]

        # Extract decision scores
        if hasattr(self.pipeline, "decision_function"):
            raw_scores = self.pipeline.decision_function(X)[0]
        else:
            # Fallback if somehow not supported
            raw_scores = [0.0] * len(self.classes_)

        scores_dict = {
            cls_name: float(score) for cls_name, score in zip(self.classes_, raw_scores)
        }

        # Calculate margin (highest - second highest)
        sorted_scores = sorted(list(raw_scores), reverse=True)
        margin = 0.0
        if len(sorted_scores) >= 2:
            margin = float(sorted_scores[0] - sorted_scores[1])

        return {
            "predicted_label": str(y_pred),
            "decision_scores": scores_dict,
            "decision_margin": margin,
            "model_name": self.model_name
        }
