"""Robustness and leakage-investigation utilities."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Label-word masking
# ---------------------------------------------------------------------------

_LABEL_WORDS = re.compile(
    r"\b(bugs?|documentation|docs?|docstrings?|enhancements?|features?|rfc)\b",
    re.IGNORECASE,
)

MASK_TOKEN = "[MASKED]"


def mask_label_words(text: str) -> str:
    """Replace obvious target-revealing words with MASK_TOKEN.

    Words masked (case-insensitive, whole-word only):
    bug, bugs, documentation, doc, docs, docstring, docstrings,
    enhancement, enhancements, feature, features, rfc.

    Note: masking is applied only inside the experimental pipeline;
    the saved dataset records are never modified.
    """
    return _LABEL_WORDS.sub(MASK_TOKEN, text)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def extract_title_only(record: dict[str, Any]) -> str:
    """Return the prefix-cleaned title."""
    return record.get("clean_title", "") or ""


def extract_body_only(record: dict[str, Any]) -> str:
    """Return the prefix-cleaned body."""
    return record.get("clean_body", "") or ""


def extract_combined(record: dict[str, Any]) -> str:
    """Return the full combined_text (title + body)."""
    return record.get("combined_text", "") or ""


# ---------------------------------------------------------------------------
# Prefix-free subset selection
# ---------------------------------------------------------------------------


def is_prefix_free(record: dict[str, Any]) -> bool:
    """Return True when the record's title was not modified by prefix removal."""
    return record.get("clean_title", "") == record.get("raw_title", "")


def prefix_free_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to records whose title had no category prefix removed."""
    result = [r for r in records if is_prefix_free(r)]
    if not result:
        raise ValueError("Prefix-free subset is empty — all records had prefixes.")
    return result


# ---------------------------------------------------------------------------
# Text-length buckets
# ---------------------------------------------------------------------------

LENGTH_BUCKETS = [
    (0, 300, "short"),
    (300, 1000, "medium"),
    (1000, 5000, "long"),
    (5000, int(1e9), "very_long"),
]


def assign_length_bucket(combined_length: int) -> str:
    """Assign a text-length bucket label."""
    for lo, hi, label in LENGTH_BUCKETS:
        if lo <= combined_length < hi:
            return label
    return "very_long"


def group_by_length_bucket(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records into text-length buckets."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        bucket = assign_length_bucket(rec.get("combined_length", 0))
        groups[bucket].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Year-based grouping
# ---------------------------------------------------------------------------


def group_by_year(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records by their created_at year."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        year = rec.get("created_at", "")[:4]
        groups[year].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def find_exact_duplicates(
    train_texts: list[str],
    val_texts: list[str],
    train_ids: list[int],
    val_ids: list[int],
) -> list[dict[str, Any]]:
    """Find validation texts that appear verbatim in training."""
    train_lookup: dict[str, int] = {}
    for text, issue_id in zip(train_texts, train_ids):
        norm = text.strip().lower()
        train_lookup[norm] = issue_id

    duplicates = []
    for text, issue_id in zip(val_texts, val_ids):
        norm = text.strip().lower()
        if norm in train_lookup:
            duplicates.append(
                {
                    "val_issue_id": issue_id,
                    "train_issue_id": train_lookup[norm],
                    "similarity": 1.0,
                }
            )
    return duplicates


def find_near_duplicates(
    train_matrix: Any,
    val_matrix: Any,
    train_ids: list[int],
    val_ids: list[int],
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[float, list[dict[str, Any]]]:
    """Find near-duplicate pairs using cosine similarity.

    Args:
        train_matrix: Sparse or dense TF-IDF matrix of training documents.
        val_matrix: Sparse or dense TF-IDF matrix of validation documents.
        train_ids: Issue IDs for training set.
        val_ids: Issue IDs for validation set.
        thresholds: Similarity thresholds to report at.

    Returns:
        Dict mapping each threshold to a list of matching pairs.
    """
    # Process in batches to keep memory manageable
    BATCH = 100
    results: dict[float, list[dict[str, Any]]] = {t: [] for t in thresholds}

    for start in range(0, val_matrix.shape[0], BATCH):
        end = min(start + BATCH, val_matrix.shape[0])
        batch = val_matrix[start:end]
        sims = cosine_similarity(batch, train_matrix)  # shape: (batch, n_train)

        for i, row_sims in enumerate(sims):
            val_idx = start + i
            best_train_idx = int(np.argmax(row_sims))
            best_sim = float(row_sims[best_train_idx])
            for thresh in thresholds:
                if best_sim >= thresh:
                    results[thresh].append(
                        {
                            "val_issue_id": val_ids[val_idx],
                            "train_issue_id": train_ids[best_train_idx],
                            "similarity": round(best_sim, 4),
                        }
                    )

    return results


# ---------------------------------------------------------------------------
# Decision-margin analysis
# ---------------------------------------------------------------------------


def get_decision_margins(
    pipeline: Any,
    X: list[str],
) -> np.ndarray | None:
    """Return decision function scores for models that support it."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "decision_function"):
        return None
    X_transformed = pipeline.named_steps["tfidf"].transform(X)
    return clf.decision_function(X_transformed)


def margin_from_scores(decision_scores: np.ndarray) -> np.ndarray:
    """Compute margin as (best score - second best score) for each sample."""
    if decision_scores.ndim == 1:
        # Binary classifier — just use absolute value
        return np.abs(decision_scores)
    sorted_scores = np.sort(decision_scores, axis=1)[:, ::-1]
    return sorted_scores[:, 0] - sorted_scores[:, 1]


# ---------------------------------------------------------------------------
# Misclassification extraction
# ---------------------------------------------------------------------------


def extract_misclassified(
    records: list[dict[str, Any]],
    y_true: list[str],
    y_pred: list[str],
    margins: np.ndarray | None,
    n_per_class: int = 15,
) -> list[dict[str, Any]]:
    """Return up to n_per_class misclassified examples per true class.

    Sorted by confidence (descending) when margins are available.
    """
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, (rec, true, pred) in enumerate(zip(records, y_true, y_pred)):
        if true != pred:
            entry = {
                "issue_id": rec.get("issue_id"),
                "true_label": true,
                "predicted_label": pred,
                "clean_title": rec.get("clean_title", "")[:120],
                "combined_length": rec.get("combined_length", 0),
                "margin": float(margins[i]) if margins is not None else None,
                "created_at": rec.get("created_at", ""),
            }
            by_class[true].append(entry)

    # Sort by descending margin (most confident wrong predictions first)
    result = []
    for true_class, entries in by_class.items():
        if margins is not None:
            entries = sorted(entries, key=lambda x: -(x["margin"] or 0))
        result.extend(entries[:n_per_class])

    return result
