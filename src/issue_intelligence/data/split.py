"""
src/issue_intelligence/data/split.py
====================================
Contains logic for splitting issue records into train, validation, and test sets.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def temporal_split(
    records: list[dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split records chronologically by created_at date.

    Args:
        records: List of dictionaries, must contain 'created_at'.
        train_frac: Fraction of data for training (oldest issues).
        val_frac: Fraction of data for validation.

    Returns:
        (train_records, val_records, test_records)
    """
    if not records:
        return [], [], []

    # Sort records by created_at ascending (oldest first)
    # Filter out records without created_at, though they should all have it
    valid_records = [r for r in records if r.get("created_at")]
    missing = len(records) - len(valid_records)
    if missing > 0:
        logger.warning(
            "Ignored %d records missing 'created_at' in temporal split.",
            missing,
        )

    sorted_records = sorted(valid_records, key=lambda x: x["created_at"])

    n_total = len(sorted_records)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_records = sorted_records[:n_train]
    val_records = sorted_records[n_train:n_train + n_val]
    test_records = sorted_records[n_train + n_val:]

    logger.info(
        "Temporal split: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_records),
        len(val_records),
        len(test_records),
        n_total,
    )

    return train_records, val_records, test_records


def stratified_random_split(
    records: list[dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split records randomly while preserving the proportion of target classes.

    Args:
        records: List of dictionaries, must contain 'target'.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_records, val_records, test_records)
    """
    if not records:
        return [], [], []

    targets = [r.get("target", "Unknown") for r in records]

    # First split: Train vs Temp (Validation + Test)
    temp_frac = 1.0 - train_frac

    try:
        train_records, temp_records, _, temp_targets = train_test_split(
            records,
            targets,
            test_size=temp_frac,
            random_state=seed,
            stratify=targets,
        )

        # Second split: Validation vs Test
        # We need the proportion of val relative to temp
        val_rel_frac = val_frac / temp_frac

        val_records, test_records = train_test_split(
            temp_records,
            test_size=(1.0 - val_rel_frac),
            random_state=seed,
            stratify=temp_targets,
        )
    except ValueError as e:
        logger.error(
            "Stratified split failed (likely too few samples per class): %s",
            e,
        )
        raise

    logger.info(
        "Stratified random split: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_records),
        len(val_records),
        len(test_records),
        len(records),
    )

    return train_records, val_records, test_records
