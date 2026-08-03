"""Dataset splitting module."""

from __future__ import annotations

import collections
import json
import random
from typing import Any


def _get_date_string(record: dict[str, Any]) -> str:
    """Extract YYYY-MM-DD from created_at."""
    return record["created_at"][:10]


def build_temporal_split(
    records: list[dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records chronologically into train, validation, and test.

    Ensures that issues created on the same calendar day (UTC) are kept in the
    same split, preventing temporal leakage from a single day's discussion.

    Returns:
        (train_records, val_records, test_records)
    """
    if not records:
        return [], [], []

    # Sort strictly chronologically by created_at, then by issue_number for stability
    sorted_records = sorted(records, key=lambda x: (x["created_at"], x["issue_number"]))

    n = len(sorted_records)
    idx_70 = int(n * train_frac)
    idx_85 = int(n * (train_frac + val_frac))

    def _adjust_idx(target_idx: int) -> int:
        if target_idx <= 0 or target_idx >= n:
            return target_idx
        # The last element going into the left split is at target_idx - 1.
        # We ensure no elements with the same date are left behind in the right split.
        base_date = _get_date_string(sorted_records[target_idx - 1])
        idx = target_idx
        # Scan forward to include all items on the same day
        while idx < n and _get_date_string(sorted_records[idx]) == base_date:
            idx += 1
        return idx

    train_end = _adjust_idx(idx_70)
    val_end = _adjust_idx(idx_85)

    # Fallback to avoid empty splits if dataset is too small
    if train_end >= n and n > 2:
        train_end = n - 2
        val_end = n - 1

    return (
        sorted_records[:train_end],
        sorted_records[train_end:val_end],
        sorted_records[val_end:],
    )


def build_stratified_random_split(
    records: list[dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records randomly but stratified by target class.

    Returns:
        (train_records, val_records, test_records)
    """
    if not records:
        return [], [], []

    rng = random.Random(seed)

    # Group by class
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        by_class[rec["target"]].append(rec)

    train, val, test = [], [], []

    for _, cls_records in sorted(by_class.items()):
        # Sort deterministically before shuffling to ensure reproducible splits
        cls_records.sort(key=lambda x: x["issue_number"])
        rng.shuffle(cls_records)

        n = len(cls_records)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        # At least one example per class per split if possible
        if n >= 3:
            if train_end == 0:
                train_end = 1
            if val_end == train_end:
                val_end = train_end + 1
            if val_end == n:
                val_end = n - 1

        train.extend(cls_records[:train_end])
        val.extend(cls_records[train_end:val_end])
        test.extend(cls_records[val_end:])

    # Shuffle the final combined splits to interleave classes
    for split in (train, val, test):
        split.sort(key=lambda x: x["issue_number"])
        rng.shuffle(split)

    return train, val, test


def verify_split_integrity(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    original_count: int,
    check_temporal: bool = False,
) -> None:
    """Validate that the splits are correct and do not overlap."""
    train_ids = {r["issue_number"] for r in train}
    val_ids = {r["issue_number"] for r in val}
    test_ids = {r["issue_number"] for r in test}

    if train_ids & val_ids:
        raise ValueError("Overlap between train and validation")
    if train_ids & test_ids:
        raise ValueError("Overlap between train and test")
    if val_ids & test_ids:
        raise ValueError("Overlap between validation and test")

    total_ids = len(train_ids) + len(val_ids) + len(test_ids)
    if total_ids != original_count:
        raise ValueError(
            f"Total IDs ({total_ids}) != original count ({original_count})"
        )

    total_records = len(train) + len(val) + len(test)
    if total_records != original_count:
        raise ValueError(
            f"Total records ({total_records}) != original count ({original_count})"
        )

    # Check all classes present (if records exist)
    for split_name, split in zip(["train", "val", "test"], [train, val, test]):
        if not split:
            continue
        classes = {r["target"] for r in split}
        for cls in ["Bug", "Documentation", "Enhancement"]:
            if cls not in classes and len(split) >= 3:
                raise ValueError(f"{cls} missing from {split_name}")

    if check_temporal and train and val and test:
        max_train = max(r["created_at"] for r in train)
        min_val = min(r["created_at"] for r in val)
        max_val = max(r["created_at"] for r in val)
        min_test = min(r["created_at"] for r in test)
        if max_train > min_val:
            raise ValueError(
                f"Temporal violation: max train {max_train} > min val {min_val}"
            )
        if max_val > min_test:
            raise ValueError(
                f"Temporal violation: max val {max_val} > min test {min_test}"
            )


def generate_manifest(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    split_type: str,
) -> dict[str, Any]:
    """Generate metadata manifest for a split."""

    def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"count": 0, "classes": {}}
        classes = collections.Counter(r["target"] for r in records)
        return {
            "count": len(records),
            "classes": dict(classes.most_common()),
            "earliest": min(r["created_at"] for r in records),
            "latest": max(r["created_at"] for r in records),
        }

    total_records = len(train) + len(val) + len(test)

    return {
        "split_type": split_type,
        "total_records": total_records,
        "train": _stats(train),
        "validation": _stats(val),
        "test": _stats(test),
        "percentages": {
            "train": round(100 * len(train) / max(1, total_records), 1),
            "validation": round(100 * len(val) / max(1, total_records), 1),
            "test": round(100 * len(test) / max(1, total_records), 1),
        },
    }


def save_jsonl(records: list[dict[str, Any]], filepath: str) -> None:
    """Save records to JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
