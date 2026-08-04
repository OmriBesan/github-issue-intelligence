import json
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from issue_intelligence.retrieval.index import build_index
from issue_intelligence.retrieval.service import RetrievalService

# A dummy vectorizer to mock TF-IDF behavior for tests
class DummyVectorizer:
    def __init__(self):
        self.vocabulary_ = {"hello": 0, "world": 1, "bug": 2}
        
    def transform(self, raw_documents):
        # Just create random sparse matrix based on number of documents
        n = len(raw_documents)
        data = np.ones(n)
        row = np.arange(n)
        col = np.zeros(n, dtype=int)
        
        # for OOV tests, if a doc is strictly "OOV", return zero matrix row
        for i, doc in enumerate(raw_documents):
            if "OOV" in doc:
                data[i] = 0.0
                
        # make shape (n, 3)
        mat = csr_matrix((data, (row, col)), shape=(n, 3))
        # Ensure it has .nnz method reflecting non-zeros
        mat.eliminate_zeros()
        return mat

@pytest.fixture
def synthetic_data_files(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    
    # Train has 2 records, one is duplicate with Val
    train_data = [
        {"issue_id": 1, "title": "hello bug", "body": "world", "target": "Bug"},
        {"issue_id": 2, "title": "another", "body": "issue", "target": "Enhancement"},
    ]
    # Val has 2 records, ID 1 is a duplicate and should be ignored
    val_data = [
        {"issue_id": 1, "title": "hello bug", "body": "world", "target": "Bug"},
        {"issue_id": 3, "title": "OOV", "body": "OOV", "target": "Documentation"},
    ]
    
    with open(train_path, "w") as f:
        for r in train_data:
            f.write(json.dumps(r) + "\n")
            
    with open(val_path, "w") as f:
        for r in val_data:
            f.write(json.dumps(r) + "\n")
            
    return train_path, val_path

def test_build_index_deduplication(synthetic_data_files):
    train_path, val_path = synthetic_data_files
    vec = DummyVectorizer()
    
    # Expecting exactly 3 unique records (IDs 1, 2, 3)
    payload = build_index(train_path, val_path, vec, expected_count=3)
    
    assert payload["artifact_metadata"]["indexed_issue_count"] == 3
    assert payload["matrix"].shape[0] == 3
    assert len(payload["metadata_list"]) == 3
    
    ids = [m["issue_id"] for m in payload["metadata_list"]]
    assert ids == [1, 2, 3]

def test_build_index_count_mismatch(synthetic_data_files):
    train_path, val_path = synthetic_data_files
    vec = DummyVectorizer()
    
    with pytest.raises(ValueError, match="Corpus size mismatch"):
        build_index(train_path, val_path, vec, expected_count=5)

@pytest.fixture
def retrieval_service(synthetic_data_files, tmp_path):
    train_path, val_path = synthetic_data_files
    vec = DummyVectorizer()
    payload = build_index(train_path, val_path, vec, expected_count=3)
    
    # Customize the matrix to specific values to test deterministic sorting and scores
    # Doc 1: score 0.8
    # Doc 2: score 0.8
    # Doc 3: OOV (all zeros)
    
    # When query comes, dummy vectorizer will output [1, 0, 0] for normal query
    # So if matrix is:
    # Row 0 (id 1): [0.8, 0, 0]
    # Row 1 (id 2): [0.8, 0, 0]
    # Row 2 (id 3): [0.0, 0, 0]
    
    payload["matrix"] = csr_matrix([
        [0.8, 0, 0],
        [0.8, 0, 0],
        [0.0, 0, 0]
    ])
    
    import joblib
    artifact_path = tmp_path / "sim.joblib"
    joblib.dump(payload, artifact_path)
    
    service = RetrievalService(artifact_path)
    service.load_artifact()
    return service

def test_retrieval_service_deterministic_sort(retrieval_service):
    # Both ID 1 and ID 2 have score 0.8 against query
    results = retrieval_service.search("hello", "world")
    
    assert len(results) == 2
    assert results[0]["issue_id"] == 1
    assert results[0]["similarity_score"] == 0.8
    
    assert results[1]["issue_id"] == 2
    assert results[1]["similarity_score"] == 0.8

def test_retrieval_service_oov(retrieval_service):
    # Query with "OOV" will cause DummyVectorizer to return all zeros
    results = retrieval_service.search("OOV", "OOV")
    assert len(results) == 0

def test_retrieval_service_exclude_id(retrieval_service):
    results = retrieval_service.search("hello", "world", exclude_issue_id=1)
    
    assert len(results) == 1
    assert results[0]["issue_id"] == 2

def test_retrieval_service_label_filter(retrieval_service):
    # ID 2 is Enhancement
    results = retrieval_service.search("hello", "world", label_filter="Enhancement")
    
    assert len(results) == 1
    assert results[0]["issue_id"] == 2
    assert results[0]["target_label"] == "Enhancement"
