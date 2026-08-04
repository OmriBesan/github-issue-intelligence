"""Tests for transformer dataset utilities.

These tests use only synthetic data and do NOT download any model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from issue_intelligence.data.transformer_dataset import (
    LABEL2ID,
    LABELS_ORDER,
    build_hf_dataset,
    compute_truncation_stats,
    decode_label,
    encode_label,
    validate_records,
)

# ---------------------------------------------------------------------------
# 1. Label encoding order
# ---------------------------------------------------------------------------


def test_label_encoding_order() -> None:
    assert encode_label("Bug") == 0
    assert encode_label("Documentation") == 1
    assert encode_label("Enhancement") == 2


def test_labels_order_consistent_with_label2id() -> None:
    for i, label in enumerate(LABELS_ORDER):
        assert LABEL2ID[label] == i


def test_decode_label_roundtrip() -> None:
    for label in LABELS_ORDER:
        assert decode_label(encode_label(label)) == label


# ---------------------------------------------------------------------------
# 2. Dataset creation from combined_text
# ---------------------------------------------------------------------------


def _make_records(n: int = 6) -> list[dict]:
    labels = ["Bug", "Documentation", "Enhancement"]
    return [
        {
            "issue_id": i + 1,
            "combined_text": f"Issue text number {i} about something important",
            "target": labels[i % 3],
            "original_labels": ["should not be used"],
            "html_url": "http://example.com",
        }
        for i in range(n)
    ]


def _make_mock_tokenizer(max_length: int = 256) -> MagicMock:
    """Return a mock tokenizer that produces predictable token ids."""

    def tokenize(
        texts,
        truncation=True,
        max_length=256,
        padding=False,
        add_special_tokens=True,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]
        results = {
            "input_ids": [[1, 2, 3, 4, 5] for _ in texts],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in texts],
        }
        return results

    mock = MagicMock()
    mock.side_effect = tokenize
    mock.__call__ = tokenize
    return mock


# ---------------------------------------------------------------------------
# 3. Missing text raises a clear error
# ---------------------------------------------------------------------------


def test_missing_combined_text_raises() -> None:
    records = [{"issue_id": 1, "combined_text": "", "target": "Bug"}]
    with pytest.raises(ValueError, match="missing or empty combined_text"):
        validate_records(records)


def test_none_text_raises() -> None:
    records = [{"issue_id": 1, "combined_text": None, "target": "Bug"}]
    with pytest.raises((ValueError, TypeError)):
        validate_records(records)


# ---------------------------------------------------------------------------
# 4. Unknown labels raise a clear error
# ---------------------------------------------------------------------------


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="Unknown label"):
        encode_label("SomeOtherLabel")


def test_unknown_label_in_records_raises() -> None:
    records = [{"combined_text": "text", "target": "BuildCI"}]
    with pytest.raises(ValueError, match="Unknown label"):
        validate_records(records)


# ---------------------------------------------------------------------------
# 5. Tokenization produces expected fields
# ---------------------------------------------------------------------------


def test_tokenization_produces_input_ids_and_attention_mask() -> None:
    """Verify build_hf_dataset creates expected columns with real tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers library not installed")

    records = _make_records(6)
    # Use bert-base-uncased: WordPiece tokenizer, no sentencepiece required,
    # already cached in HuggingFace hub for most environments.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = build_hf_dataset(records, tokenizer, max_length=32)
    assert "label" in ds.column_names
    assert "input_ids" in ds.column_names
    assert "attention_mask" in ds.column_names
    # text column should be removed after tokenization
    assert "text" not in ds.column_names


# ---------------------------------------------------------------------------
# 6. Truncation statistics are correct
# ---------------------------------------------------------------------------


def test_truncation_stats_counts_correctly() -> None:
    # Create records with text that contains the index at a known position
    records = [
        {
            "issue_id": i,
            "combined_text": f"record_{i}_text",
            "target": ["Bug", "Documentation", "Enhancement"][i % 3],
        }
        for i in range(9)
    ]

    mock_tok = MagicMock()

    def fake_tokenize(text, truncation=False, add_special_tokens=True):
        # Simulate: records 0,1,2 get 300 tokens; rest get 100
        # Parse index from "record_{i}_text"
        idx = int(text.split("_")[1])
        length = 300 if idx < 3 else 100
        return {"input_ids": list(range(length))}

    mock_tok.side_effect = fake_tokenize

    stats = compute_truncation_stats(records, mock_tok, max_length=256)
    assert stats["n_truncated"] == 3
    assert stats["pct_truncated"] == pytest.approx(33.33, abs=0.1)
    assert stats["max_tokens"] == 300
    assert stats["min_tokens"] == 100


# ---------------------------------------------------------------------------
# 7. Dataset does not expose metadata as model input
# ---------------------------------------------------------------------------


def test_metadata_not_in_dataset_columns() -> None:
    try:
        import datasets as _ds_mod  # noqa: F401
    except ImportError:
        pytest.skip("datasets library not installed")

    records = _make_records(6)
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": [[1, 2, 3]] * 6,
        "attention_mask": [[1, 1, 1]] * 6,
    }
    mock_tok.side_effect = None

    ds = build_hf_dataset(records, mock_tok, max_length=32)
    forbidden = {"original_labels", "html_url", "issue_id", "raw_title", "raw_body"}
    assert forbidden.isdisjoint(set(ds.column_names))


# ---------------------------------------------------------------------------
# 8. Metrics use the fixed class order
# ---------------------------------------------------------------------------


def test_metrics_use_fixed_label_order() -> None:
    from issue_intelligence.evaluation.metrics import compute_metrics

    y_true = ["Bug", "Documentation", "Enhancement"]
    y_pred = ["Bug", "Documentation", "Enhancement"]
    result = compute_metrics(y_true, y_pred, labels=LABELS_ORDER)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert set(result["per_class"].keys()) == set(LABELS_ORDER)


# ---------------------------------------------------------------------------
# 12. Random seeds are configured (in transformer.py defaults)
# ---------------------------------------------------------------------------


def test_default_seed_is_defined() -> None:
    from issue_intelligence.models.transformer import DEFAULT_SEED

    assert isinstance(DEFAULT_SEED, int)
    assert DEFAULT_SEED == 42


# ---------------------------------------------------------------------------
# 13. Output result schema is complete
# ---------------------------------------------------------------------------


def test_output_schema_keys() -> None:
    """The result dict must contain all required metric keys."""
    from issue_intelligence.evaluation.metrics import compute_metrics

    y_true = ["Bug", "Bug", "Documentation"]
    y_pred = ["Bug", "Enhancement", "Documentation"]
    result = compute_metrics(y_true, y_pred, labels=LABELS_ORDER)

    required = {"macro_f1", "weighted_f1", "accuracy", "per_class"}
    assert required.issubset(result.keys())
    for label in LABELS_ORDER:
        assert label in result["per_class"]
        assert set(result["per_class"][label].keys()) >= {"precision", "recall", "f1"}


# ---------------------------------------------------------------------------
# 15. Empty datasets raise clear errors
# ---------------------------------------------------------------------------


def test_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_records([])
