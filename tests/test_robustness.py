"""Tests for robustness evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from issue_intelligence.evaluation.robustness import (
    MASK_TOKEN,
    assign_length_bucket,
    extract_body_only,
    extract_misclassified,
    extract_title_only,
    find_exact_duplicates,
    group_by_length_bucket,
    group_by_year,
    is_prefix_free,
    mask_label_words,
    prefix_free_subset,
)

# ---------------------------------------------------------------------------
# 1. Title-only extraction
# ---------------------------------------------------------------------------


def test_title_only() -> None:
    rec = {
        "clean_title": "My title",
        "clean_body": "My body",
        "combined_text": "t\n\nb",
    }
    assert extract_title_only(rec) == "My title"


def test_title_only_missing_key() -> None:
    assert extract_title_only({}) == ""


# ---------------------------------------------------------------------------
# 2. Body-only extraction
# ---------------------------------------------------------------------------


def test_body_only() -> None:
    rec = {"clean_title": "My title", "clean_body": "My body"}
    assert extract_body_only(rec) == "My body"


def test_body_only_missing_key() -> None:
    assert extract_body_only({}) == ""


# ---------------------------------------------------------------------------
# 3. Label-word masking — case insensitive
# ---------------------------------------------------------------------------


def test_masking_bug_lowercase() -> None:
    assert MASK_TOKEN in mask_label_words("This is a bug report")


def test_masking_bug_uppercase() -> None:
    assert MASK_TOKEN in mask_label_words("BUG: something wrong")


def test_masking_documentation() -> None:
    assert MASK_TOKEN in mask_label_words("Update the documentation")


def test_masking_docs() -> None:
    assert MASK_TOKEN in mask_label_words("Fix the docs typo")


def test_masking_docstring() -> None:
    assert MASK_TOKEN in mask_label_words("The docstring is wrong")


def test_masking_enhancement() -> None:
    assert MASK_TOKEN in mask_label_words("Add this enhancement")


def test_masking_feature() -> None:
    assert MASK_TOKEN in mask_label_words("New feature request")


def test_masking_rfc() -> None:
    assert MASK_TOKEN in mask_label_words("RFC: add new feature")


def test_masking_does_not_affect_unrelated_words() -> None:
    text = "Fix the TypeError in numpy array"
    assert mask_label_words(text) == text


# ---------------------------------------------------------------------------
# 4. Masking does not modify the input record
# ---------------------------------------------------------------------------


def test_masking_does_not_modify_record() -> None:
    original = "There is a bug in this code"
    rec = {"combined_text": original}
    _ = mask_label_words(rec["combined_text"])
    # Record must be unchanged
    assert rec["combined_text"] == original


# ---------------------------------------------------------------------------
# 5. Prefix-free subset selection
# ---------------------------------------------------------------------------


def test_prefix_free_is_true_when_titles_match() -> None:
    rec = {"clean_title": "Fix memory error", "raw_title": "Fix memory error"}
    assert is_prefix_free(rec) is True


def test_prefix_free_is_false_when_prefix_removed() -> None:
    rec = {"clean_title": "memory error", "raw_title": "BUG: memory error"}
    assert is_prefix_free(rec) is False


def test_prefix_free_subset_filters_correctly() -> None:
    records = [
        {"clean_title": "Fix memory", "raw_title": "Fix memory"},
        {"clean_title": "memory error", "raw_title": "BUG: memory error"},
    ]
    subset = prefix_free_subset(records)
    assert len(subset) == 1
    assert subset[0]["clean_title"] == "Fix memory"


def test_prefix_free_empty_raises() -> None:
    records = [
        {"clean_title": "memory error", "raw_title": "BUG: memory error"},
    ]
    with pytest.raises(ValueError, match="empty"):
        prefix_free_subset(records)


# ---------------------------------------------------------------------------
# 6. Exact duplicate detection
# ---------------------------------------------------------------------------


def test_exact_duplicate_found() -> None:
    train_texts = ["this is a bug report", "enhancement request here"]
    val_texts = ["this is a bug report", "different text"]
    dupes = find_exact_duplicates(train_texts, val_texts, [1, 2], [101, 102])
    assert len(dupes) == 1
    assert dupes[0]["val_issue_id"] == 101


def test_no_exact_duplicates() -> None:
    train_texts = ["train text one", "train text two"]
    val_texts = ["val text one", "val text two"]
    dupes = find_exact_duplicates(train_texts, val_texts, [1, 2], [101, 102])
    assert dupes == []


# ---------------------------------------------------------------------------
# 7. Text-length bucket assignment
# ---------------------------------------------------------------------------


def test_short_bucket() -> None:
    assert assign_length_bucket(100) == "short"


def test_medium_bucket() -> None:
    assert assign_length_bucket(500) == "medium"


def test_long_bucket() -> None:
    assert assign_length_bucket(2000) == "long"


def test_very_long_bucket() -> None:
    assert assign_length_bucket(10000) == "very_long"


def test_group_by_length_bucket() -> None:
    records = [
        {"combined_length": 100},
        {"combined_length": 500},
        {"combined_length": 2000},
    ]
    groups = group_by_length_bucket(records)
    assert "short" in groups
    assert "medium" in groups
    assert "long" in groups


# ---------------------------------------------------------------------------
# 8. Year-based grouping
# ---------------------------------------------------------------------------


def test_group_by_year() -> None:
    records = [
        {"created_at": "2022-10-01T00:00:00Z"},
        {"created_at": "2023-05-01T00:00:00Z"},
        {"created_at": "2023-11-01T00:00:00Z"},
        {"created_at": "2024-01-01T00:00:00Z"},
    ]
    groups = group_by_year(records)
    assert "2022" in groups
    assert "2023" in groups
    assert "2024" in groups
    assert len(groups["2023"]) == 2


# ---------------------------------------------------------------------------
# 9. Misclassified-example extraction
# ---------------------------------------------------------------------------


def test_misclassified_extraction() -> None:
    records = [
        {
            "issue_id": 1,
            "clean_title": "bad prediction",
            "combined_length": 100,
            "created_at": "",
        },
        {
            "issue_id": 2,
            "clean_title": "correct prediction",
            "combined_length": 200,
            "created_at": "",
        },
    ]
    y_true = ["Bug", "Enhancement"]
    y_pred = ["Enhancement", "Enhancement"]

    misclassified = extract_misclassified(
        records, y_true, y_pred, margins=None, n_per_class=5
    )
    assert len(misclassified) == 1
    assert misclassified[0]["issue_id"] == 1
    assert misclassified[0]["true_label"] == "Bug"
    assert misclassified[0]["predicted_label"] == "Enhancement"


def test_misclassified_sorted_by_margin() -> None:
    records = [
        {"issue_id": 1, "clean_title": "A", "combined_length": 100, "created_at": ""},
        {"issue_id": 2, "clean_title": "B", "combined_length": 100, "created_at": ""},
    ]
    y_true = ["Bug", "Bug"]
    y_pred = ["Enhancement", "Enhancement"]
    margins = np.array([0.5, 2.0])

    result = extract_misclassified(
        records, y_true, y_pred, margins=margins, n_per_class=5
    )
    # Higher margin should come first
    assert result[0]["issue_id"] == 2
