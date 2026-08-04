"""Evaluate a saved transformer model on temporal validation.

Usage:
  .venv\\Scripts\\python scripts\\evaluate_transformer.py \\
      --model-dir models/transformer/best \\
      --splits-dir data/processed/splits \\
      --results-dir reports/results \\
      --figures-dir reports/figures/transformer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.transformer_dataset import (
    LABELS_ORDER,
    build_hf_dataset,
    decode_label,
    validate_records,
)
from issue_intelligence.evaluation.metrics import compute_metrics


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    recs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    if not recs:
        raise ValueError(f"Empty file: {filepath}")
    return recs


def main() -> None:
    _setup_logging()
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    # Load temporal val only — NEVER test
    logging.info("Loading temporal validation …")
    val_recs = load_jsonl(args.splits_dir / "temporal" / "validation.jsonl")
    validate_records(val_recs, "val")
    y_val_true = [r["target"] for r in val_recs]

    # Load saved model
    logging.info(f"Loading model from {args.model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))

    val_ds = build_hf_dataset(val_recs, tokenizer, args.max_length, "val")

    eval_args = TrainingArguments(
        output_dir=str(args.results_dir / "_eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=eval_args)

    t0 = time.time()
    pred_output = trainer.predict(val_ds)
    inference_time = time.time() - t0

    y_pred_ids = np.argmax(pred_output.predictions, axis=-1)
    y_pred = [decode_label(int(p)) for p in y_pred_ids]

    metrics = compute_metrics(y_val_true, y_pred, labels=LABELS_ORDER)
    logging.info(f"Temporal val Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"Inference time: {inference_time:.3f}s")

    result = {
        "temporal_val_macro_f1": metrics["macro_f1"],
        "temporal_val_accuracy": metrics["accuracy"],
        "temporal_val_per_class": metrics["per_class"],
        "inference_time_s": round(inference_time, 3),
    }

    out = args.results_dir / "transformer_eval_only.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
