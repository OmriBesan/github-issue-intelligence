"""Transformer model utilities for issue classification."""

from __future__ import annotations

import logging
from typing import Any

from issue_intelligence.data.transformer_dataset import (
    ID2LABEL,
    LABEL2ID,
    LABELS_ORDER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model and training defaults
# ---------------------------------------------------------------------------

# Selected checkpoint: google/bert_uncased_L-2_H-128_A-2  ("BERT-Tiny")
#
# Reason for NOT using distilbert-base-uncased (the preferred checkpoint):
#   Hardware: CPU-only Windows 11, 3.8 GB free RAM, no CUDA GPU.
#   Timing estimate: DistilBERT (66M params, 6 layers) requires ~14.5s/step
#   at batch=16, seq_len=256. With 250 steps/epoch this gives ~60 min/epoch,
#   or ~180 min for 3 epochs — unreasonable for an experimental comparison.
#
# BERT-Tiny (4.4M params, 2 layers, hidden=128) is the official Google
# checkpoint from https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
# It uses the standard BERT WordPiece tokenizer (no sentencepiece required).
# Expected speed: ~20× faster than DistilBERT → ~9 min for 3 epochs on CPU.
#
# Expected tradeoff:
#   BERT-Tiny has 15× fewer parameters and shallower attention (2 vs 6 layers).
#   It captures less contextual information. Its classification F1 on downstream
#   tasks is typically 5-15 pp below DistilBERT on easy tasks and more on hard
#   tasks. This is an acceptable tradeoff given the hardware constraint.
#   The goal is NOT to beat DistilBERT but to determine if ANY transformer
#   beats the tuned LinearSVC (0.9087 Macro F1).
DEFAULT_CHECKPOINT = "google/bert_uncased_L-2_H-128_A-2"
FALLBACK_CHECKPOINT = "google/bert_uncased_L-4_H-256_A-4"  # BERT-Mini, 11M params

DEFAULT_MAX_LENGTH = 256
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42


def get_class_weights(
    records: list[dict[str, Any]],
) -> list[float]:
    """Compute inverse-frequency class weights for the training records.

    Returns weights in LABELS_ORDER = [Bug, Documentation, Enhancement].
    """
    from collections import Counter

    counts = Counter(r["target"] for r in records)
    total = sum(counts.values())
    n_classes = len(LABELS_ORDER)
    weights = [
        total / (n_classes * counts.get(label, 1))
        for label in LABELS_ORDER
    ]
    return weights


def load_model_and_tokenizer(
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> tuple[Any, Any]:
    """Load a sequence-classification model and its tokenizer.

    Args:
        checkpoint: HuggingFace model identifier.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading tokenizer from {checkpoint!r}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    logger.info(f"Loading model from {checkpoint!r}")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tokenizer


def count_parameters(model: Any) -> dict[str, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
