import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Stage 6: Semantic Search (Duplicate Detection)\n\nIn this notebook, we embed our training dataset using SentenceTransformers and use it as a retrieval corpus to find similar issues."),

    nbf.v4.new_code_cell("""import json
import sys
from pathlib import Path

sys.path.append("../src")
from issue_intelligence.models.retrieval import IssueRetriever

DATA_DIR = Path("../data/processed")
"""),

    nbf.v4.new_markdown_cell("## 1. Load Data"),

    nbf.v4.new_code_cell("""def load_data(file_path):
    X = []
    if not file_path.exists():
        return X
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            X.append(r.get("combined_text", ""))
    return X

corpus = load_data(DATA_DIR / "scikit-learn_issues_model_stratified_train.jsonl")
test_queries = load_data(DATA_DIR / "scikit-learn_issues_model_stratified_test.jsonl")

print(f"Corpus size: {len(corpus)}")
print(f"Test queries size: {len(test_queries)}")
"""),

    nbf.v4.new_markdown_cell("## 2. Initialize Retriever and Embed Corpus"),

    nbf.v4.new_code_cell("""retriever = IssueRetriever(model_name="all-MiniLM-L6-v2")
print("Embedding corpus... (This might take a moment the first time)")
retriever.embed_corpus(corpus)
print("Corpus embedded successfully!")
"""),

    nbf.v4.new_markdown_cell("## 3. Test Retrieval"),

    nbf.v4.new_code_cell("""# Pick a sample query from the test set
sample_query = test_queries[0]
print("QUERY:")
print("-" * 40)
print(sample_query)
print("-" * 40)

print("\\nTOP 3 MOST SIMILAR ISSUES IN CORPUS:")
results = retriever.search(sample_query, top_k=3)
for i, res in enumerate(results):
    print(f"\\n[{i+1}] Score: {res['score']:.4f}")
    print(res['text'][:200] + "...")
""")
]

nb['cells'] = cells
with open('notebooks/05_semantic_search.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
