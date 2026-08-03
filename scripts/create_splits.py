"""CLI script to create dataset splits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.splitting import (
    build_stratified_random_split,
    build_temporal_split,
    generate_manifest,
    save_jsonl,
    verify_split_integrity,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_split_stats(name: str, split: list[dict], original_total: int) -> None:
    print(f"\n  --- {name.upper()} ---")
    if not split:
        print("  (Empty)")
        return

    count = len(split)
    pct = 100 * count / original_total

    from collections import Counter

    classes = Counter(r["target"] for r in split)

    earliest = min(r["created_at"] for r in split)
    latest = max(r["created_at"] for r in split)

    print(f"  Records    : {count:,} ({pct:.1f}%)")
    print(f"  Date range : {earliest[:10]} to {latest[:10]}")
    for cls in ["Bug", "Enhancement", "Documentation"]:
        cls_count = classes.get(cls, 0)
        cls_pct = 100 * cls_count / max(1, count)
        print(f"    {cls:<15}: {cls_count:>5,} ({cls_pct:.1f}%)")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument(
        "--input", required=True, type=Path, help="Input processed JSONL"
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--train", type=float, default=0.70, help="Train fraction")
    parser.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not args.input.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logging.info(f"Loading {args.input}...")
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    logging.info(f"Loaded {total:,} records.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- TEMPORAL SPLIT ---
    logging.info("Building temporal split...")
    t_train, t_val, t_test = build_temporal_split(records, args.train, args.val)
    verify_split_integrity(t_train, t_val, t_test, total, check_temporal=True)

    temporal_dir = args.out_dir / "temporal"
    temporal_dir.mkdir(exist_ok=True)
    save_jsonl(t_train, str(temporal_dir / "train.jsonl"))
    save_jsonl(t_val, str(temporal_dir / "validation.jsonl"))
    save_jsonl(t_test, str(temporal_dir / "test.jsonl"))

    t_manifest = generate_manifest(t_train, t_val, t_test, "temporal")
    with open(temporal_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(t_manifest, f, indent=2)

    print("\n====================================================================")
    print("  TEMPORAL SPLIT (PRIMARY)")
    print("====================================================================")
    _print_split_stats("Train", t_train, total)
    _print_split_stats("Validation", t_val, total)
    _print_split_stats("Test", t_test, total)

    # --- STRATIFIED RANDOM SPLIT ---
    logging.info("Building stratified random split...")
    r_train, r_val, r_test = build_stratified_random_split(
        records, args.train, args.val, args.seed
    )
    verify_split_integrity(r_train, r_val, r_test, total, check_temporal=False)

    random_dir = args.out_dir / "random"
    random_dir.mkdir(exist_ok=True)
    save_jsonl(r_train, str(random_dir / "train.jsonl"))
    save_jsonl(r_val, str(random_dir / "validation.jsonl"))
    save_jsonl(r_test, str(random_dir / "test.jsonl"))

    r_manifest = generate_manifest(r_train, r_val, r_test, "stratified_random")
    r_manifest["seed"] = args.seed
    with open(random_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(r_manifest, f, indent=2)

    print("\n====================================================================")
    print("  STRATIFIED RANDOM SPLIT (SECONDARY / COMPARISON ONLY)")
    print("====================================================================")
    _print_split_stats("Train", r_train, total)
    _print_split_stats("Validation", r_val, total)
    _print_split_stats("Test", r_test, total)


if __name__ == "__main__":
    main()
