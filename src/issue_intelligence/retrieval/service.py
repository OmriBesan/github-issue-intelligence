import logging
from pathlib import Path
from typing import Any

import joblib

from issue_intelligence.data.preprocessing import (
    build_combined_text,
    clean_body,
    clean_title,
)

logger = logging.getLogger(__name__)

class RetrievalService:
    """Service to load retrieval artifact and execute TF-IDF cosine similarity search."""

    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.is_ready = False
        self.vectorizer = None
        self.matrix = None
        self.metadata_list = []
        self.artifact_metadata = {}

    def load_artifact(self) -> None:
        if not self.artifact_path.exists():
            logger.error(f"Retrieval artifact not found: {self.artifact_path}")
            self.is_ready = False
            return

        try:
            artifact = joblib.load(self.artifact_path)

            self.artifact_metadata = artifact.get("artifact_metadata", {})
            if self.artifact_metadata.get("artifact_version") != "1.0":
                raise ValueError("Unsupported or missing artifact_version.")

            self.vectorizer = artifact.get("vectorizer")
            if self.vectorizer is None:
                raise ValueError("Vectorizer missing from artifact.")

            self.matrix = artifact.get("matrix")
            if self.matrix is None:
                raise ValueError("Sparse matrix missing from artifact.")

            self.metadata_list = artifact.get("metadata_list", [])

            # Validation
            expected_count = self.artifact_metadata.get("indexed_issue_count", 0)
            if self.matrix.shape[0] != expected_count:
                raise ValueError("Matrix row count does not match indexed_issue_count.")
            if len(self.metadata_list) != expected_count:
                raise ValueError("Metadata length does not match indexed_issue_count.")

            # Check required fields exist in first item
            if expected_count > 0:
                req_fields = {"issue_number", "title", "target_label", "created_at", "url"}
                if not req_fields.issubset(self.metadata_list[0].keys()):
                    raise ValueError("Metadata items are missing required fields.")

            self.is_ready = True
            logger.info(f"Retrieval artifact loaded successfully. {expected_count} issues indexed.")

        except Exception as e:
            logger.error(f"Failed to load corrupt retrieval artifact from {self.artifact_path}: {e}")
            self.is_ready = False
            self.vectorizer = None
            self.matrix = None
            self.metadata_list = []

    def search(
        self,
        title: str | None,
        body: str | None,
        top_k: int = 5,
        label_filter: str | None = None,
        exclude_issue_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the index using exact cosine similarity over TF-IDF vectors."""
        if not self.is_ready:
            raise RuntimeError("Retrieval index is not loaded.")

        clean_t, _ = clean_title(title)
        clean_b = clean_body(body)
        combined = build_combined_text(clean_t, clean_b)

        if not combined:
            return []

        # Transform query
        q_vec = self.vectorizer.transform([combined])

        # OOV check: if vector is all zeros, similarity is zero everywhere
        if q_vec.nnz == 0:
            return []

        # The sparse matrix has l2-normalized rows by default from TfidfVectorizer,
        # so cosine similarity is just the dot product.
        # q_vec is also l2-normalized.
        # similarities shape: (1, n_samples)
        similarities = (self.matrix @ q_vec.T).toarray().flatten()

        results = []
        for idx in range(len(similarities)):
            score = float(similarities[idx])

            meta = self.metadata_list[idx]
            if label_filter and meta.get("target_label") != label_filter:
                continue

            if exclude_issue_id is not None and meta.get("issue_id") == exclude_issue_id:
                continue

            # Allow items with zero score? The user said: "If OOV, return empty. Do not return arbitrary zero-similarity issues."
            if score == 0.0:
                continue

            res = meta.copy()
            res["similarity_score"] = score
            results.append(res)

        # Deterministic sorting: similarity desc, then issue_number asc
        results.sort(key=lambda x: (-x["similarity_score"], x["issue_number"]))

        return results[:top_k]
