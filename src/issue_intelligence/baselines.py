import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class MajorityClassBaseline:
    """
    A simple baseline model that always predicts the most frequent class
    observed during training.
    """

    def __init__(self) -> None:
        self.majority_class_: str | None = None

    def fit(self, X: Any, y: list[str]) -> "MajorityClassBaseline":
        """
        Learn the majority class from the training labels.

        Args:
            X: Ignored, included for scikit-learn compatibility.
            y: List of training labels.
        """
        if not y:
            raise ValueError("Training labels y cannot be empty.")

        counts = Counter(y)
        self.majority_class_ = counts.most_common(1)[0][0]
        logger.debug("Majority class determined as: %s", self.majority_class_)
        return self

    def predict(self, X: list[Any]) -> list[str]:
        """
        Predict the majority class for all samples.

        Args:
            X: List of samples to predict.

        Returns:
            List of predicted labels (all being the majority class).
        """
        if self.majority_class_ is None:
            raise RuntimeError("Model must be fitted before predict is called.")
        return [self.majority_class_] * len(X)


class RuleBasedBaseline:
    """
    A baseline model that uses simple keyword heuristics to classify issues.
    If no keywords match, it falls back to a default class (usually Bug).
    """

    def __init__(self, default_class: str = "Bug") -> None:
        self.default_class = default_class
        # Simple heuristic keywords (case-insensitive)
        self.doc_keywords = ["doc", "documentation", "readme", "typo", "spelling"]
        self.enhancement_keywords = ["add", "feature", "enhancement", "support", "new"]
        self.bug_keywords = ["bug", "error", "fail", "crash", "fix", "issue"]

    def fit(self, X: Any, y: Any) -> "RuleBasedBaseline":
        """
        Rule-based model requires no training.

        Args:
            X: Ignored.
            y: Ignored.
        """
        return self

    def predict(self, X: list[str]) -> list[str]:
        """
        Predict class based on keyword matching in the text.

        Args:
            X: List of issue texts (e.g. combined title and body).

        Returns:
            List of predicted labels.
        """
        predictions = []
        for text in X:
            text_lower = str(text).lower()

            # Simple scoring based on keyword hits
            scores = {"Documentation": 0, "Enhancement": 0, "Bug": 0}

            for kw in self.doc_keywords:
                if kw in text_lower:
                    scores["Documentation"] += 1
            for kw in self.enhancement_keywords:
                if kw in text_lower:
                    scores["Enhancement"] += 1
            for kw in self.bug_keywords:
                if kw in text_lower:
                    scores["Bug"] += 1

            # Find the max score
            max_score = max(scores.values())

            if max_score == 0:
                predictions.append(self.default_class)
            else:
                # If tied, we just take the first one that matches the max in this order
                # Docs usually have specific keywords, so prefer that if tied
                if scores["Documentation"] == max_score:
                    predictions.append("Documentation")
                elif scores["Enhancement"] == max_score:
                    predictions.append("Enhancement")
                else:
                    predictions.append("Bug")

        return predictions
