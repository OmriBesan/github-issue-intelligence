"""
tests/test_split.py
"""

import pytest

from issue_intelligence.data.split import stratified_random_split, temporal_split


@pytest.fixture
def mock_records():
    return [
        {"id": 1, "created_at": "2023-01-01T10:00:00Z", "target": "Bug"},
        {"id": 2, "created_at": "2023-01-02T10:00:00Z", "target": "Enhancement"},
        {"id": 3, "created_at": "2023-01-03T10:00:00Z", "target": "Bug"},
        {"id": 4, "created_at": "2023-01-04T10:00:00Z", "target": "Documentation"},
        {"id": 5, "created_at": "2023-01-05T10:00:00Z", "target": "Bug"},
        {"id": 6, "created_at": "2023-01-06T10:00:00Z", "target": "Enhancement"},
        {"id": 7, "created_at": "2023-01-07T10:00:00Z", "target": "Documentation"},
        {"id": 8, "created_at": "2023-01-08T10:00:00Z", "target": "Enhancement"},
        {"id": 9, "created_at": "2023-01-09T10:00:00Z", "target": "Bug"},
        {"id": 10, "created_at": "2023-01-10T10:00:00Z", "target": "Bug"},
    ]


def test_temporal_split_sizes(mock_records):
    train_rec, val_rec, test_rec = temporal_split(
        mock_records, train_frac=0.6, val_frac=0.2
    )
    assert len(train_rec) == 6
    assert len(val_rec) == 2
    assert len(test_rec) == 2


def test_temporal_split_order(mock_records):
    # Reverse the list so it's not sorted
    reversed_records = list(reversed(mock_records))
    train_rec, val_rec, test_rec = temporal_split(
        reversed_records, train_frac=0.6, val_frac=0.2
    )

    # Train should have older records than Val, Val older than Test
    assert max(r["created_at"] for r in train_rec) < min(
        r["created_at"] for r in val_rec
    )
    assert max(r["created_at"] for r in val_rec) < min(
        r["created_at"] for r in test_rec
    )


def test_temporal_split_empty():
    train_rec, val_rec, test_rec = temporal_split([])
    assert len(train_rec) == 0
    assert len(val_rec) == 0
    assert len(test_rec) == 0


def test_stratified_random_split_sizes():
    records = []
    for i in range(10):
        records.append({"id": i, "target": "Bug"})
    for i in range(10, 20):
        records.append({"id": i, "target": "Enhancement"})
    for i in range(20, 30):
        records.append({"id": i, "target": "Documentation"})

    train_rec, val_rec, test_rec = stratified_random_split(
        records, train_frac=0.6, val_frac=0.2
    )
    assert 17 <= len(train_rec) <= 19
    assert 5 <= len(val_rec) <= 7
    assert 5 <= len(test_rec) <= 7


def test_stratified_random_split_proportions():
    # For a larger set, test the proportions.
    records = []
    for i in range(50):
        records.append({"id": i, "target": "Bug"})
    for i in range(50, 80):
        records.append({"id": i, "target": "Enhancement"})
    for i in range(80, 100):
        records.append({"id": i, "target": "Documentation"})

    train_rec, val_rec, test_rec = stratified_random_split(
        records, train_frac=0.7, val_frac=0.15
    )

    # Verify sizes
    assert 69 <= len(train_rec) <= 71
    assert 14 <= len(val_rec) <= 16
    assert 14 <= len(test_rec) <= 16

    # Verify stratification in train set (roughly 50% bug, 30% enhancement, 20% doc)
    targets_train = [r["target"] for r in train_rec]
    assert 34 <= targets_train.count("Bug") <= 36
    assert 20 <= targets_train.count("Enhancement") <= 22
    assert 13 <= targets_train.count("Documentation") <= 15


def test_stratified_random_split_empty():
    train_rec, val_rec, test_rec = stratified_random_split([])
    assert len(train_rec) == 0
    assert len(val_rec) == 0
    assert len(test_rec) == 0
