"""Transformer dataset utilities for issue classification."""

from __future__ import annotations

from typing import Any

LABEL2ID: dict[str, int] = {
    "Bug": 0,
    "Documentation": 1,
    "Enhancement": 2,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
LABELS_ORDER = ["Bug", "Documentation", "Enhancement"]


def encode_label(label: str) -> int:
    """Return integer label id for a string class name.

    Args:
        label: One of "Bug", "Documentation", "Enhancement".

    Returns:
        Integer id (0, 1, or 2).

    Raises:
        ValueError: If the label is not one of the three valid classes.
    """
    if label not in LABEL2ID:
        raise ValueError(
            f"Unknown label: {label!r}. Valid labels: {list(LABEL2ID.keys())}"
        )
    return LABEL2ID[label]


def decode_label(label_id: int) -> str:
    """Return string class name for an integer label id."""
    if label_id not in ID2LABEL:
        raise ValueError(
            f"Unknown label id: {label_id}. Valid ids: {list(ID2LABEL.keys())}"
        )
    return ID2LABEL[label_id]


def validate_records(
    records: list[dict[str, Any]],
    split_name: str = "dataset",
) -> None:
    """Check that records contain required fields and valid labels.

    Args:
        records: List of record dicts.
        split_name: Name for error messages.

    Raises:
        ValueError: If records are empty, missing fields, or have unknown labels.
    """
    if not records:
        raise ValueError(f"{split_name} is empty.")

    for i, rec in enumerate(records):
        if "combined_text" not in rec or not rec["combined_text"]:
            raise ValueError(
                f"{split_name} record {i} (issue_id={rec.get('issue_id')}) "
                f"has missing or empty combined_text."
            )
        if "target" not in rec:
            raise ValueError(f"{split_name} record {i} has no 'target' field.")
        encode_label(rec["target"])  # raises ValueError on unknown label


def build_hf_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 256,
    split_name: str = "dataset",
) -> Any:
    """Build a HuggingFace Dataset from record dicts.

    Only combined_text and label are included as model inputs.
    No metadata (issue_id, html_url, original_labels, etc.) is exposed.

    Args:
        records: Issue records from JSONL files.
        tokenizer: A HuggingFace PreTrainedTokenizer.
        max_length: Maximum token length (longer texts are truncated).
        split_name: Name for validation error messages.

    Returns:
        A HuggingFace Dataset with columns: input_ids, attention_mask, label.
    """
    from datasets import Dataset  # imported here to keep module importable without HF

    validate_records(records, split_name)

    texts = [r["combined_text"] for r in records]
    labels = [encode_label(r["target"]) for r in records]

    raw_ds = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # handled by DataCollatorWithPadding at training time
        )

    tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized


def compute_truncation_stats(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 256,
) -> dict[str, Any]:
    """Compute token-length statistics and truncation rate.

    Args:
        records: Issue records (should be training split).
        tokenizer: HuggingFace tokenizer.
        max_length: Threshold above which a record is considered truncated.

    Returns:
        Dict with: min, max, mean, median, p90, p95, p99 token lengths,
        n_truncated, pct_truncated.
    """
    import statistics

    lengths = []
    for rec in records:
        ids = tokenizer(
            rec["combined_text"],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        lengths.append(len(ids))

    n_truncated = sum(1 for ln in lengths if ln > max_length)
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    return {
        "n_total": n,
        "n_truncated": n_truncated,
        "pct_truncated": round(100 * n_truncated / n, 2) if n > 0 else 0.0,
        "min_tokens": min(lengths_sorted),
        "max_tokens": max(lengths_sorted),
        "mean_tokens": round(statistics.mean(lengths), 1),
        "median_tokens": statistics.median(lengths),
        "p90_tokens": lengths_sorted[int(0.90 * n)],
        "p95_tokens": lengths_sorted[int(0.95 * n)],
        "p99_tokens": lengths_sorted[int(0.99 * n)],
    }
