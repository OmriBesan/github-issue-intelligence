import argparse
import logging
from pathlib import Path

import joblib

from issue_intelligence.retrieval.index import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF similar-issue retrieval artifact.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/splits/temporal/train.jsonl")
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/processed/splits/temporal/validation.jsonl")
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/classical/final_linear_svc.joblib")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/retrieval/similar_issues.joblib")
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Final model not found: {args.model_path}")

    logger.info(f"Loading final model from {args.model_path}")
    pipeline = joblib.load(args.model_path)

    vectorizer = pipeline.named_steps.get("tfidf")
    if not vectorizer:
        raise ValueError("Could not find 'tfidf' step in the final model pipeline.")

    logger.info("Building index...")
    artifact_payload = build_index(
        train_path=args.train_file,
        val_path=args.val_file,
        vectorizer=vectorizer,
        expected_class_distribution={"Bug": 1873, "Enhancement": 1790, "Documentation": 1191}
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving artifact to {args.output_path}")
    joblib.dump(artifact_payload, args.output_path)

    logger.info("Retrieval artifact build complete.")

if __name__ == "__main__":
    main()
