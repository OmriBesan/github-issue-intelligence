import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Stage 5: Transformer Models\n\nHere we fine-tune DistilBERT on our dataset and compare its performance to the baselines."),

    nbf.v4.new_code_cell("""import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

sys.path.append("../src")
from issue_intelligence.models.transformer import prepare_dataset, build_transformer_trainer

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("../data/processed")
"""),

    nbf.v4.new_markdown_cell("## 1. Load Data"),

    nbf.v4.new_code_cell("""def load_data(file_path):
    X, y = [], []
    if not file_path.exists():
        return X, y
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            X.append(r.get("combined_text", ""))
            y.append(r.get("target", "Unknown"))
    return X, y

X_train, y_train = load_data(DATA_DIR / "scikit-learn_issues_model_stratified_train.jsonl")
X_test, y_test = load_data(DATA_DIR / "scikit-learn_issues_model_stratified_test.jsonl")

le = LabelEncoder()
train_dataset, le = prepare_dataset(X_train, y_train, le)
test_dataset, le = prepare_dataset(X_test, y_test, le)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")
print(f"Classes: {le.classes_}")
"""),

    nbf.v4.new_markdown_cell("## 2. Train DistilBERT"),

    nbf.v4.new_code_cell("""trainer, tokenizer = build_transformer_trainer(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    model_name="distilbert-base-uncased",
    num_labels=len(le.classes_),
    epochs=2,
    batch_size=8,
)

print("Starting Fine-Tuning...")
trainer.train()
"""),

    nbf.v4.new_markdown_cell("## 3. Evaluation"),

    nbf.v4.new_code_cell("""print("Evaluating on Test Set...")
eval_results = trainer.evaluate()
print(f"Transformer Macro F1: {eval_results['eval_f1']:.4f}")

# Get raw predictions
predictions = trainer.predict(trainer.eval_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

y_true_labels = le.inverse_transform(predictions.label_ids)
y_pred_labels = le.inverse_transform(preds)

print("\\nClassification Report:\\n")
print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
""")
]

nb['cells'] = cells
with open('notebooks/04_transformer_models.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
