import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from issue_intelligence.data.preprocessing import build_combined_text, clean_body, clean_title

logger = logging.getLogger(__name__)

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def build_index(
    train_path: Path,
    val_path: Path,
    vectorizer: Any,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Build the retrieval index using train and validation records.
    
    Args:
        train_path: Path to the training split JSONL.
        val_path: Path to the validation split JSONL.
        vectorizer: The fitted TF-IDF vectorizer.
        expected_count: Optional integer to validate the final deduplicated count.
        
    Returns:
        A dictionary containing the artifact payload to be saved.
    """
    logger.info("Loading training and validation records...")
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    
    # Deduplicate by issue_id to prevent any internal leakage or overlaps
    seen_ids = set()
    unique_records = []
    for r in train_records + val_records:
        issue_id = r["issue_id"]
        if issue_id not in seen_ids:
            seen_ids.add(issue_id)
            unique_records.append(r)
            
    total_indexed = len(unique_records)
    logger.info(f"Indexed {total_indexed} unique records.")
    
    if expected_count is not None and total_indexed != expected_count:
        raise ValueError(
            f"Corpus size mismatch. Expected {expected_count}, got {total_indexed}."
        )
        
    logger.info("Preprocessing text and collecting metadata...")
    combined_texts = []
    metadata_list = []
    class_counts: dict[str, int] = {}
    
    dates = []
    
    for r in unique_records:
        t, _ = clean_title(r.get("raw_title"))
        b = clean_body(r.get("raw_body"))
        combined = build_combined_text(t, b)
        combined_texts.append(combined)
        
        target = r.get("target")
        class_counts[target] = class_counts.get(target, 0) + 1
        
        c_at = r.get("created_at")
        if c_at:
            dates.append(c_at)
            
        metadata_list.append({
            "issue_id": r["issue_id"],
            "title": r.get("raw_title", ""),
            "target_label": target,
            "created_at": c_at,
            "url": r.get("html_url")
        })
        
    logger.info("Transforming texts into TF-IDF sparse matrix...")
    sparse_matrix = vectorizer.transform(combined_texts)
    
    vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else 0
    
    dates.sort()
    corpus_min = dates[0] if dates else None
    corpus_max = dates[-1] if dates else None
    
    artifact_metadata = {
        "artifact_version": "1.0",
        "indexed_issue_count": total_indexed,
        "matrix_shape": sparse_matrix.shape,
        "vocabulary_size": vocab_size,
        "corpus_date_min": corpus_min,
        "corpus_date_max": corpus_max,
        "class_distribution": class_counts,
        "source_splits": [train_path.name, val_path.name],
        "build_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    return {
        "vectorizer": vectorizer,
        "matrix": sparse_matrix,
        "metadata_list": metadata_list,
        "artifact_metadata": artifact_metadata,
    }
