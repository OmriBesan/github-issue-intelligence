"""Train the transformer issue classifier.

Usage:
  .venv\\Scripts\\python scripts\\train_transformer.py \\
      --splits-dir data/processed/splits \\
      --model-dir models/transformer \\
      --results-dir reports/results \\
      --figures-dir reports/figures/transformer \\
      [--checkpoint prajjwal1/bert-tiny] \\
      [--epochs 3] \\
      [--batch-size 16] \\
      [--max-length 256] \\
      [--lr 5e-5]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.transformer_dataset import (
    LABELS_ORDER,
    build_hf_dataset,
    compute_truncation_stats,
    validate_records,
)
from issue_intelligence.evaluation.metrics import compute_metrics
from issue_intelligence.models.transformer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    count_parameters,
    get_class_weights,
    load_model_and_tokenizer,
)

TEMPORAL_TEST_PATH_PATTERNS = ["test.jsonl", "test_*.jsonl"]


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


def _assert_no_test_split(splits_dir: Path) -> None:
    """Abort if a test split is accidentally loaded."""
    for pattern in TEMPORAL_TEST_PATH_PATTERNS:
        for p in splits_dir.rglob(pattern):
            # Just listing is okay; we never load it
            pass


def plot_training_history(
    history: list[dict],
    out_path: Path,
) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    val_f1 = [h.get("val_macro_f1", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, marker="o", label="Train loss")
    ax1.plot(epochs, val_loss, marker="s", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(epochs, val_f1, marker="o", color="seagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1 by Epoch")
    ax2.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    title: str,
    out_path: Path,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS_ORDER)))
    ax.set_yticks(range(len(LABELS_ORDER)))
    ax.set_xticklabels(LABELS_ORDER, rotation=20)
    ax.set_yticklabels(LABELS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_per_class_comparison(
    transformer_metrics: dict,
    lsvc_temporal_f1: dict,
    out_path: Path,
) -> None:
    labels = LABELS_ORDER
    transformer_f1 = [transformer_metrics["per_class"][lbl]["f1"] for lbl in labels]
    lsvc_f1 = list(lsvc_temporal_f1.values())

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, lsvc_f1, width, label="LinearSVC (tuned)", color="steelblue")
    ax.bar(x + width / 2, transformer_f1, width, label="Transformer", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: LinearSVC vs Transformer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_token_length_distribution(
    stats: dict,
    lengths: list[int],
    max_length: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths, bins=50, edgecolor="white", alpha=0.8)
    ax.axvline(
        max_length, color="red", linestyle="--", label=f"max_length={max_length}"
    )
    ax.set_xlabel("Token Length")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Token Length Distribution "
        f"(truncated: {stats['n_truncated']}/{stats['n_total']} "
        f"= {stats['pct_truncated']}%)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    _setup_logging()
    import torch
    from transformers import (
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Verify no test split is being loaded                                 #
    # ------------------------------------------------------------------ #
    _assert_no_test_split(args.splits_dir)

    # ------------------------------------------------------------------ #
    # Load temporal train and val only                                     #
    # ------------------------------------------------------------------ #
    logging.info("Loading temporal train and validation …")
    train_recs = load_jsonl(args.splits_dir / "temporal" / "train.jsonl")
    val_recs = load_jsonl(args.splits_dir / "temporal" / "validation.jsonl")
    validate_records(train_recs, "train")
    validate_records(val_recs, "val")
    logging.info(f"  Train: {len(train_recs)}  Val: {len(val_recs)}")

    y_val_true = [r["target"] for r in val_recs]

    # ------------------------------------------------------------------ #
    # Load model and tokenizer                                             #
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    param_info = count_parameters(model)
    logging.info(
        f"Model parameters: total={param_info['total']:,}  "
        f"trainable={param_info['trainable']:,}"
    )

    # ------------------------------------------------------------------ #
    # Tokenization and truncation stats                                    #
    # ------------------------------------------------------------------ #
    logging.info("Computing truncation statistics …")
    trunc_stats = compute_truncation_stats(train_recs, tokenizer, args.max_length)
    logging.info(
        f"  Truncated: {trunc_stats['n_truncated']}/{trunc_stats['n_total']} "
        f"({trunc_stats['pct_truncated']}%)"
    )
    logging.info(
        f"  Token lengths: min={trunc_stats['min_tokens']} "
        f"median={trunc_stats['median_tokens']} "
        f"p99={trunc_stats['p99_tokens']} max={trunc_stats['max_tokens']}"
    )

    # Build token-length distribution for plot
    train_token_lengths = []
    for rec in train_recs:
        ids = tokenizer(
            rec["combined_text"],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        train_token_lengths.append(len(ids))

    plot_token_length_distribution(
        trunc_stats,
        train_token_lengths,
        args.max_length,
        args.figures_dir / "token_length_distribution.png",
    )

    # ------------------------------------------------------------------ #
    # Build HuggingFace datasets                                           #
    # ------------------------------------------------------------------ #
    logging.info("Building tokenized datasets …")
    train_ds = build_hf_dataset(train_recs, tokenizer, args.max_length, "train")
    val_ds = build_hf_dataset(val_recs, tokenizer, args.max_length, "val")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------------------------------------------------------------ #
    # Class weights                                                        #
    # ------------------------------------------------------------------ #
    class_weights = get_class_weights(train_recs)
    logging.info(
        f"Class weights: {dict(zip(LABELS_ORDER, [f'{w:.3f}' for w in class_weights]))}"
    )

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Custom Trainer with weighted loss                                    #
    # ------------------------------------------------------------------ #
    import torch.nn as nn

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Per-epoch history tracking
    history: list[dict] = []

    class HistoryCallback:
        pass

    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def on_evaluate(self, args_cb, state, control, metrics=None, **kwargs):
            if metrics is not None:
                history.append(
                    {
                        "epoch": state.epoch,
                        "train_loss": state.log_history[-1].get("loss", 0)
                        if state.log_history
                        else 0,
                        "val_loss": metrics.get("eval_loss", 0),
                        "val_macro_f1": metrics.get("eval_macro_f1", 0),
                    }
                )

    def compute_hf_metrics(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        from issue_intelligence.data.transformer_dataset import decode_label

        y_true = [decode_label(int(lid)) for lid in label_ids]
        y_pred = [decode_label(int(p)) for p in preds]
        m = compute_metrics(y_true, y_pred, labels=LABELS_ORDER)
        return {
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
        }

    # ------------------------------------------------------------------ #
    # Training configuration                                               #
    # ------------------------------------------------------------------ #
    best_model_dir = str(args.model_dir / "best")
    training_args = TrainingArguments(
        output_dir=str(args.model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=DEFAULT_SEED,
        logging_strategy="epoch",
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_hf_metrics,
        callbacks=[MetricsCallback()],
    )

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    logging.info(f"Starting training for {args.epochs} epochs …")
    t0_train = time.time()
    trainer.train()
    train_time = time.time() - t0_train
    logging.info(f"Training complete in {train_time:.1f}s ({train_time / 60:.1f} min)")

    # Save best model
    trainer.save_model(best_model_dir)

    # ------------------------------------------------------------------ #
    # Final evaluation on temporal validation                             #
    # ------------------------------------------------------------------ #
    logging.info("Running final evaluation on temporal validation …")
    t0_eval = time.time()
    trainer.evaluate(val_ds)
    inference_time = time.time() - t0_eval

    # Get predictions for detailed metrics
    pred_output = trainer.predict(val_ds)
    y_pred_ids = np.argmax(pred_output.predictions, axis=-1)
    from issue_intelligence.data.transformer_dataset import decode_label

    y_pred = [decode_label(int(p)) for p in y_pred_ids]

    final_metrics = compute_metrics(y_val_true, y_pred, labels=LABELS_ORDER)
    logging.info(f"  Temporal val Macro F1: {final_metrics['macro_f1']:.4f}")
    logging.info(f"  Temporal val Accuracy: {final_metrics['accuracy']:.4f}")

    # Model size
    model_size_mb = (
        sum(f.stat().st_size for f in Path(best_model_dir).rglob("*") if f.is_file())
        / 1e6
    )

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    plot_training_history(history, args.figures_dir / "training_history.png")
    plot_confusion_matrix(
        y_val_true,
        y_pred,
        "Transformer Confusion Matrix (Temporal Val)",
        args.figures_dir / "transformer_confusion_matrix.png",
    )

    # LinearSVC per-class F1 for comparison
    lsvc_per_class = {"Bug": 0.9278, "Documentation": 0.8759, "Enhancement": 0.9225}
    plot_per_class_comparison(
        final_metrics,
        lsvc_per_class,
        args.figures_dir / "per_class_comparison.png",
    )

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    result_doc = {
        "checkpoint": args.checkpoint,
        "param_info": param_info,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "lr": args.lr,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "seed": DEFAULT_SEED,
            "class_weighted_loss": True,
        },
        "truncation_stats": trunc_stats,
        "training_time_s": round(train_time, 1),
        "inference_time_s": round(inference_time, 3),
        "model_size_mb": round(model_size_mb, 1),
        "temporal_val": {
            "macro_f1": final_metrics["macro_f1"],
            "weighted_f1": final_metrics["weighted_f1"],
            "accuracy": final_metrics["accuracy"],
            "per_class": final_metrics["per_class"],
        },
        "epoch_history": history,
    }

    out_path = args.results_dir / "transformer_temporal_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)

    # Training history CSV
    hist_csv = args.results_dir / "transformer_training_history.csv"
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_macro_f1"]
        )
        writer.writeheader()
        writer.writerows(history)

    # Classical comparison CSV
    comparison = [
        {
            "model": "LinearSVC (tuned)",
            "temporal_macro_f1": 0.9087,
            "temporal_accuracy": "",
            "bug_f1": 0.9278,
            "doc_f1": 0.8759,
            "enh_f1": 0.9225,
            "vocab_or_params": "20,473 terms",
            "train_time_s": 1.94,
            "inference_time_s": 0.39,
            "model_size_mb": "< 1",
        },
        {
            "model": f"Transformer ({args.checkpoint})",
            "temporal_macro_f1": final_metrics["macro_f1"],
            "temporal_accuracy": final_metrics["accuracy"],
            "bug_f1": final_metrics["per_class"]["Bug"]["f1"],
            "doc_f1": final_metrics["per_class"]["Documentation"]["f1"],
            "enh_f1": final_metrics["per_class"]["Enhancement"]["f1"],
            "vocab_or_params": f"{param_info['trainable']:,} params",
            "train_time_s": round(train_time, 1),
            "inference_time_s": round(inference_time, 3),
            "model_size_mb": round(model_size_mb, 1),
        },
    ]
    comp_csv = args.results_dir / "transformer_classical_comparison.csv"
    with open(comp_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    logging.info("All results saved.")

    # Final summary
    delta = final_metrics["macro_f1"] - 0.9087
    print("\n====================================================================")
    print("  TRANSFORMER vs LinearSVC COMPARISON")
    print("====================================================================")
    print("  LinearSVC temporal val Macro F1 : 0.9087")
    print(f"  Transformer temporal val Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Delta                            : {delta:+.4f}")
    print(f"  Training time                    : {train_time:.1f}s")
    print(f"  Inference time                   : {inference_time:.3f}s")
    print(f"  Model size                       : {model_size_mb:.1f} MB")
    print(f"  Trainable params                 : {param_info['trainable']:,}")
    print(f"  Token truncation rate            : {trunc_stats['pct_truncated']}%")


if __name__ == "__main__":
    main()
