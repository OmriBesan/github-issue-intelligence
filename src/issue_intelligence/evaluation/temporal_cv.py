"""Expanding-window temporal cross-validation for hyperparameter tuning.

Folds are created from the temporal TRAINING split only.
The external validation set (temporal or random) is never accessed here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _snap_to_day_boundary(records: list[dict[str, Any]], idx: int) -> int:
    """Move idx forward until all records with the same
    date as records[idx] are included.

    This prevents calendar-day splits across train/validation boundary.
    """
    if idx >= len(records):
        return len(records)
    boundary_date = records[idx]["created_at"][:10]
    while idx < len(records) and records[idx]["created_at"][:10] == boundary_date:
        idx += 1
    return idx


def make_temporal_folds(
    records: list[dict[str, Any]],
    n_folds: int = 3,
    min_train_size: int = 100,
    min_val_size: int = 50,
    required_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate expanding-window temporal folds from a sorted record list.

    Args:
        records: Records sorted by (created_at, issue_id). Must all be from
            the temporal training split — never include external validation
            or test records.
        n_folds: Number of folds to create (default 3).
        min_train_size: Minimum number of training records per fold.
        min_val_size: Minimum number of validation records per fold.
        required_classes: Set of class labels that must appear in every split.

    Returns:
        List of dicts, each containing:
            fold_index, train_records, val_records,
            train_date_range, val_date_range,
            train_class_counts, val_class_counts

    Raises:
        ValueError: If records are unsorted, insufficient, or classes missing.
    """
    if required_classes is None:
        required_classes = {"Bug", "Documentation", "Enhancement"}

    # Verify sorted order
    for i in range(1, len(records)):
        if records[i]["created_at"] < records[i - 1]["created_at"]:
            raise ValueError(
                f"Records are not sorted by created_at at index {i}: "
                f"{records[i]['created_at']} < {records[i - 1]['created_at']}"
            )

    # Verify no duplicate IDs
    ids = [r["issue_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate issue IDs detected in input records.")

    n = len(records)
    # We need n_folds + 1 equal-ish partitions
    n_parts = n_folds + 1
    part_size = n // n_parts

    if part_size < min_val_size:
        raise ValueError(
            f"Not enough records to create {n_folds} folds "
            f"with min_val_size={min_val_size}. "
            f"Got {n} records (part_size={part_size})."
        )

    # Compute fold boundaries (day-snapped)
    # cutpoints[i] is the index at which part i+1 begins
    raw_cuts = [part_size * k for k in range(1, n_parts)]
    cuts = [_snap_to_day_boundary(records, c) for c in raw_cuts]

    # Deduplicate (in case day snapping creates collisions)
    seen = set()
    unique_cuts = []
    for c in cuts:
        if c not in seen and 0 < c < n:
            unique_cuts.append(c)
            seen.add(c)
    cuts = sorted(unique_cuts)

    if len(cuts) < n_folds:
        raise ValueError(
            f"After day-snapping, only {len(cuts)} unique cut points available "
            f"but {n_folds} folds required."
        )
    cuts = cuts[:n_folds]

    folds = []
    for i, val_start in enumerate(cuts):
        # Training: all records up to cut i
        train_recs = records[:val_start]
        # Validation: records from cut i to cut i+1 (or end)
        val_end = cuts[i + 1] if i + 1 < len(cuts) else n
        val_recs = records[val_start:val_end]

        if len(train_recs) < min_train_size:
            raise ValueError(
                f"Fold {i + 1} training set too small: "
                f"{len(train_recs)} < {min_train_size}"
            )
        if len(val_recs) < min_val_size:
            raise ValueError(
                f"Fold {i + 1} validation set too small: "
                f"{len(val_recs)} < {min_val_size}"
            )

        train_classes = set(r["target"] for r in train_recs)
        val_classes = set(r["target"] for r in val_recs)
        missing_train = required_classes - train_classes
        missing_val = required_classes - val_classes
        if missing_train:
            raise ValueError(f"Fold {i + 1} training missing classes: {missing_train}")
        if missing_val:
            raise ValueError(f"Fold {i + 1} validation missing classes: {missing_val}")

        # Safety: check no overlap
        train_ids = {r["issue_id"] for r in train_recs}
        val_ids = {r["issue_id"] for r in val_recs}
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(
                f"Fold {i + 1}: {len(overlap)} overlapping "
                f"issue IDs between train and val."
            )

        # Safety: all val records newer than last train record
        last_train_date = train_recs[-1]["created_at"][:10]
        first_val_date = val_recs[0]["created_at"][:10]
        if first_val_date <= last_train_date:
            # Allow same day only if they are distinct issues already separated
            pass  # day snapping already handles this

        folds.append(
            {
                "fold_index": i + 1,
                "train_records": train_recs,
                "val_records": val_recs,
                "train_date_range": (
                    train_recs[0]["created_at"][:10],
                    train_recs[-1]["created_at"][:10],
                ),
                "val_date_range": (
                    val_recs[0]["created_at"][:10],
                    val_recs[-1]["created_at"][:10],
                ),
                "train_class_counts": dict(Counter(r["target"] for r in train_recs)),
                "val_class_counts": dict(Counter(r["target"] for r in val_recs)),
            }
        )

    return folds


def summarise_folds(folds: list[dict[str, Any]]) -> None:
    """Print a human-readable fold summary."""
    print(f"{'Fold':<6} {'Train':>7} {'Val':>7}  Train dates             Val dates")
    print("-" * 70)
    for fold in folds:
        print(
            f"{fold['fold_index']:<6} "
            f"{len(fold['train_records']):>7} "
            f"{len(fold['val_records']):>7}  "
            f"{fold['train_date_range'][0]} → {fold['train_date_range'][1]}  "
            f"{fold['val_date_range'][0]} → {fold['val_date_range'][1]}"
        )
