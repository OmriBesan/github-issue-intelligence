import json
import sys
from pathlib import Path

import joblib

sys.path.append("src")
from issue_intelligence.models.classical import build_classical_pipeline
from issue_intelligence.models.retrieval import IssueRetriever


def main():
    data_dir = Path("data/processed")
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train and save Classical Classifier (Logistic Regression)
    print("Building Classifier Asset...")
    X_train, y_train = [], []
    train_file = data_dir / "scikit-learn_issues_model_stratified_train.jsonl"
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                X_train.append(r.get("combined_text", ""))
                y_train.append(r.get("target", "Unknown"))

    if not X_train:
        print("No training data found. Make sure you ran the dummy generator script.")
        return

    clf = build_classical_pipeline("logistic_regression")
    clf.fit(X_train, y_train)

    clf_path = model_dir / "classifier.pkl"
    joblib.dump(clf, clf_path)
    print(f"Saved classifier to {clf_path}")

    # 2. Embed corpus and save Retriever texts & embeddings
    print("Building Retriever Asset...")
    retriever = IssueRetriever(model_name="all-MiniLM-L6-v2")
    retriever.embed_corpus(X_train)

    retriever_data = {
        "corpus_texts": retriever.corpus_texts,
        "corpus_embeddings": retriever.corpus_embeddings
    }

    ret_path = model_dir / "retriever.pkl"
    joblib.dump(retriever_data, ret_path)
    print(f"Saved retriever data to {ret_path}")

if __name__ == "__main__":
    main()
