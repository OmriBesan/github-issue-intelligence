"""Tests for dataset splitting module."""

from __future__ import annotations

from typing import Any

import pytest

from issue_intelligence.data.splitting import (
    build_stratified_random_split,
    build_temporal_split,
    generate_manifest,
    verify_split_integrity,
)


def _make_record(id: int, date: str, target: str) -> dict[str, Any]:
    return {
        "issue_number": id,
        "created_at": date,
        "target": target,
        "clean_title": f"Title {id}",
        "clean_body": f"Body {id}",
    }


def test_temporal_ordering() -> None:
    records = []
    for i in range(1, 11):
        target = (
            "Bug" if i % 3 == 0 else ("Documentation" if i % 3 == 1 else "Enhancement")
        )
        records.append(_make_record(i, f"2020-01-{i:02d}T10:00:00Z", target))
    train, val, test = build_temporal_split(records, train_frac=0.7, val_frac=0.15)
    assert len(train) == 7
    assert len(val) == 1
    assert len(test) == 2
    verify_split_integrity(train, val, test, len(records), check_temporal=True)


def test_temporal_cutoff_assignment_same_day() -> None:
    records = [
        _make_record(1, "2020-01-01T10:00:00Z", "Bug"),
        _make_record(2, "2020-01-02T10:00:00Z", "Documentation"),
        # These 3 are on the same day.
        _make_record(3, "2020-01-03T01:00:00Z", "Enhancement"),
        _make_record(4, "2020-01-03T10:00:00Z", "Bug"),
        _make_record(5, "2020-01-03T23:00:00Z", "Documentation"),
        _make_record(6, "2020-01-04T10:00:00Z", "Enhancement"),
        _make_record(7, "2020-01-05T10:00:00Z", "Bug"),
    ]
    # 7 records. 30% = 2.1 -> idx 2. base date (idx 1) is "2020-01-02".
    # idx 2 is "2020-01-03". No shift needed. train gets 2.
    # 60% = 4.2 -> idx 4. base date (idx 3) is "2020-01-03".
    # idx 4 is "2020-01-03", idx 5 is "2020-01-04". So idx shifts to 5. val gets 3.
    train, val, test = build_temporal_split(records, train_frac=0.3, val_frac=0.3)

    assert len(train) == 2
    assert len(val) == 3
    assert len(test) == 2
    verify_split_integrity(train, val, test, len(records), check_temporal=True)


def test_no_overlap() -> None:
    records = []
    for i in range(1, 20):
        target = (
            "Bug" if i % 3 == 0 else ("Documentation" if i % 3 == 1 else "Enhancement")
        )
        records.append(_make_record(i, f"2020-01-{i:02d}T10:00:00Z", target))
    train, val, test = build_temporal_split(records)
    verify_split_integrity(train, val, test, len(records), check_temporal=True)

    train, val, test = build_stratified_random_split(records)
    verify_split_integrity(train, val, test, len(records), check_temporal=False)


def test_all_classes_present() -> None:
    records = []
    # 10 of each class
    for i in range(10):
        records.append(_make_record(i * 3 + 1, f"2020-01-{i + 1:02d}T10:00:00Z", "Bug"))
        records.append(
            _make_record(i * 3 + 2, f"2020-01-{i + 1:02d}T11:00:00Z", "Documentation")
        )
        records.append(
            _make_record(i * 3 + 3, f"2020-01-{i + 1:02d}T12:00:00Z", "Enhancement")
        )

    train, val, test = build_stratified_random_split(
        records, train_frac=0.7, val_frac=0.15
    )

    for split in [train, val, test]:
        classes = {r["target"] for r in split}
        assert classes == {"Bug", "Documentation", "Enhancement"}


def test_deterministic_random_split() -> None:
    records = [
        _make_record(i, f"2020-01-{i % 28 + 1:02d}T10:00:00Z", "Bug")
        for i in range(1, 100)
    ]

    t1, v1, te1 = build_stratified_random_split(records, seed=42)
    t2, v2, te2 = build_stratified_random_split(records, seed=42)

    assert [r["issue_number"] for r in t1] == [r["issue_number"] for r in t2]
    assert [r["issue_number"] for r in v1] == [r["issue_number"] for r in v2]
    assert [r["issue_number"] for r in te1] == [r["issue_number"] for r in te2]


def test_stratified_class_proportions() -> None:
    records = []
    # Imbalanced classes
    for i in range(100):
        records.append(_make_record(i, f"2020-01-{i % 28 + 1:02d}T10:00:00Z", "Bug"))
    for i in range(50):
        records.append(
            _make_record(
                100 + i, f"2020-01-{i % 28 + 1:02d}T10:00:00Z", "Documentation"
            )
        )
    for i in range(25):
        records.append(
            _make_record(150 + i, f"2020-01-{i % 28 + 1:02d}T10:00:00Z", "Enhancement")
        )

    train, val, test = build_stratified_random_split(
        records, train_frac=0.6, val_frac=0.2
    )

    # Check Bug proportion in train (~60%)
    train_bugs = len([r for r in train if r["target"] == "Bug"])
    assert train_bugs == 60

    train_docs = len([r for r in train if r["target"] == "Documentation"])
    assert train_docs == 30


def test_empty_input() -> None:
    train, val, test = build_temporal_split([])
    assert train == []
    assert val == []
    assert test == []

    train, val, test = build_stratified_random_split([])
    assert train == []
    assert val == []
    assert test == []


def test_insufficient_examples_for_one_class() -> None:
    records = [
        _make_record(1, "2020-01-01T10:00:00Z", "Bug"),
        _make_record(2, "2020-01-02T10:00:00Z", "Documentation"),
        # Enhancement only has 1 example, can't be in all 3 splits
        _make_record(3, "2020-01-03T10:00:00Z", "Enhancement"),
    ]
    train, val, test = build_stratified_random_split(records)
    verify_split_integrity(train, val, test, len(records))
    # It shouldn't crash, but won't be in all 3.
    # Enhancement will go to train because it's only 1 record.


def test_manifest_statistics() -> None:
    records = [
        _make_record(1, "2020-01-01T10:00:00Z", "Bug"),
        _make_record(2, "2020-01-02T10:00:00Z", "Documentation"),
        _make_record(3, "2020-01-03T10:00:00Z", "Enhancement"),
        _make_record(4, "2020-01-04T10:00:00Z", "Bug"),
    ]
    train = [records[0], records[1]]
    val = [records[2]]
    test = [records[3]]

    manifest = generate_manifest(train, val, test, "temporal")

    assert manifest["split_type"] == "temporal"
    assert manifest["total_records"] == 4
    assert manifest["train"]["count"] == 2
    assert manifest["train"]["classes"] == {"Bug": 1, "Documentation": 1}
    assert manifest["validation"]["classes"] == {"Enhancement": 1}


def test_verify_integrity_catches_overlap() -> None:
    records = [
        _make_record(1, "2020-01-01T10:00:00Z", "Bug"),
        _make_record(2, "2020-01-02T10:00:00Z", "Documentation"),
    ]
    with pytest.raises(ValueError, match="Overlap between train and validation"):
        verify_split_integrity([records[0]], [records[0], records[1]], [], 2)


def test_verify_integrity_catches_missing_records() -> None:
    records = [
        _make_record(1, "2020-01-01T10:00:00Z", "Bug"),
        _make_record(2, "2020-01-02T10:00:00Z", "Documentation"),
    ]
    with pytest.raises(ValueError, match="Total IDs"):
        verify_split_integrity([records[0]], [], [], 2)
