"""Tests for the temporal cross-validation module."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from issue_intelligence.evaluation.temporal_cv import make_temporal_folds


def _make_records(
    n: int, start_year: int = 2010, classes: list[str] | None = None
) -> list[dict]:
    """Create n synthetic dated records spread over several years."""
    if classes is None:
        classes = ["Bug", "Documentation", "Enhancement"]
    records = []
    base = date(start_year, 1, 1)
    for i in range(n):
        # spread 1 per week
        d = base + timedelta(weeks=i)
        records.append(
            {
                "issue_id": i + 1,
                "created_at": d.strftime("%Y-%m-%dT00:00:00Z"),
                "target": classes[i % len(classes)],
                "combined_text": f"text {i}",
            }
        )
    return records


_FOLD_KWARGS = {"n_folds": 3, "min_train_size": 50, "min_val_size": 30}


# ---------------------------------------------------------------------------
# 1. Temporal folds preserve chronological order
# ---------------------------------------------------------------------------


def test_chronological_order_preserved() -> None:
    recs = _make_records(300)
    folds = make_temporal_folds(recs, **_FOLD_KWARGS)
    for fold in folds:
        tr = fold["train_records"]
        vl = fold["val_records"]
        # last train date <= first val date
        assert tr[-1]["created_at"] <= vl[0]["created_at"]


# ---------------------------------------------------------------------------
# 2. No issue IDs overlap within a fold
# ---------------------------------------------------------------------------


def test_no_id_overlap_in_fold() -> None:
    recs = _make_records(300)
    folds = make_temporal_folds(recs, **_FOLD_KWARGS)
    for fold in folds:
        tr_ids = {r["issue_id"] for r in fold["train_records"]}
        vl_ids = {r["issue_id"] for r in fold["val_records"]}
        assert tr_ids.isdisjoint(vl_ids)


# ---------------------------------------------------------------------------
# 3. No validation record is older than its training boundary
# ---------------------------------------------------------------------------


def test_validation_records_are_newer_than_train() -> None:
    recs = _make_records(300)
    folds = make_temporal_folds(recs, **_FOLD_KWARGS)
    for fold in folds:
        last_train_date = fold["train_records"][-1]["created_at"]
        first_val_date = fold["val_records"][0]["created_at"]
        # val must start no earlier than last training date
        assert first_val_date >= last_train_date


# ---------------------------------------------------------------------------
# 4. All records come from the training split (no val/test contamination)
#    – verified by checking folds use only the input records
# ---------------------------------------------------------------------------


def test_all_records_from_input() -> None:
    recs = _make_records(300)
    input_ids = {r["issue_id"] for r in recs}
    folds = make_temporal_folds(recs, **_FOLD_KWARGS)
    for fold in folds:
        all_ids = {r["issue_id"] for r in fold["train_records"]} | {
            r["issue_id"] for r in fold["val_records"]
        }
        assert all_ids.issubset(input_ids)


# ---------------------------------------------------------------------------
# 5. All three classes appear in every fold
# ---------------------------------------------------------------------------


def test_all_classes_in_every_fold() -> None:
    recs = _make_records(300)
    folds = make_temporal_folds(recs, **_FOLD_KWARGS)
    required = {"Bug", "Documentation", "Enhancement"}
    for fold in folds:
        train_classes = {r["target"] for r in fold["train_records"]}
        val_classes = {r["target"] for r in fold["val_records"]}
        assert required.issubset(train_classes)
        assert required.issubset(val_classes)


# ---------------------------------------------------------------------------
# 6. Fold generation is deterministic
# ---------------------------------------------------------------------------


def test_deterministic_fold_generation() -> None:
    recs = _make_records(300)
    folds1 = make_temporal_folds(recs, **_FOLD_KWARGS)
    folds2 = make_temporal_folds(recs, **_FOLD_KWARGS)
    for f1, f2 in zip(folds1, folds2):
        assert len(f1["train_records"]) == len(f2["train_records"])
        assert len(f1["val_records"]) == len(f2["val_records"])
        assert f1["train_date_range"] == f2["train_date_range"]


# ---------------------------------------------------------------------------
# 7. Invalid / insufficient datasets raise clear errors
# ---------------------------------------------------------------------------


def test_unsorted_records_raise_error() -> None:
    recs = _make_records(300)
    recs_reversed = list(reversed(recs))
    with pytest.raises(ValueError, match="sorted"):
        make_temporal_folds(recs_reversed, **_FOLD_KWARGS)


def test_too_few_records_raises_error() -> None:
    recs = _make_records(10)
    with pytest.raises(ValueError):
        make_temporal_folds(recs, n_folds=3, min_val_size=50)


def test_duplicate_ids_raise_error() -> None:
    recs = _make_records(300)
    recs[0]["issue_id"] = recs[1]["issue_id"]  # create duplicate
    with pytest.raises(ValueError, match="Duplicate"):
        make_temporal_folds(recs, **_FOLD_KWARGS)


# ---------------------------------------------------------------------------
# 8. Expanding structure: each fold's training set is strictly larger
# ---------------------------------------------------------------------------


def test_expanding_window_train_sizes() -> None:
    recs = _make_records(300)
    folds = make_temporal_folds(recs, **_FOLD_KWARGS)
    train_sizes = [len(f["train_records"]) for f in folds]
    # each fold should have more training data than the previous
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1]
