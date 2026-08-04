"""API schemas and Pydantic models."""

from pydantic import BaseModel, model_validator


class PredictRequest(BaseModel):
    """Request payload for prediction."""
    title: str | None = None
    body: str | None = None

    @model_validator(mode="after")
    def check_non_empty(self) -> "PredictRequest":
        """Ensure at least one field is non-empty."""
        t = (self.title or "").strip()
        b = (self.body or "").strip()
        if not t and not b:
            raise ValueError("title and body cannot both be blank")

        # Optional: enforce max lengths
        if len(self.title or "") > 2000:
            raise ValueError("Title exceeds maximum length of 2000 characters")
        if len(self.body or "") > 100000:
            raise ValueError("Body exceeds maximum length of 100,000 characters")

        return self


class PredictResponse(BaseModel):
    """Response payload for prediction."""
    predicted_label: str
    decision_scores: dict[str, float]
    decision_margin: float
    model_name: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    retrieval_loaded: bool | None = None


class ModelInfoResponse(BaseModel):
    """Model info response."""
    model_loaded: bool
    expected_fields: list[str]
    labels: list[str] | None = None
    vocabulary_size: int | None = None
    model_type: str | None = None


class SimilarRequest(BaseModel):
    """Request payload for similar-issue retrieval."""
    title: str | None = None
    body: str | None = None
    top_k: int = 5
    label_filter: str | None = None

    @model_validator(mode="after")
    def check_non_empty(self) -> "SimilarRequest":
        """Ensure at least one field is non-empty and bounds are respected."""
        t = (self.title or "").strip()
        b = (self.body or "").strip()
        if not t and not b:
            raise ValueError("title and body cannot both be blank")

        if len(self.title or "") > 2000:
            raise ValueError("Title exceeds maximum length of 2000 characters")
        if len(self.body or "") > 100000:
            raise ValueError("Body exceeds maximum length of 100,000 characters")

        if not (1 <= self.top_k <= 10):
            raise ValueError("top_k must be between 1 and 10")

        return self


class SimilarIssueItem(BaseModel):
    """A single similar issue result."""
    issue_number: int
    issue_id: int | None = None
    title: str
    target_label: str | None = None
    created_at: str | None = None
    url: str | None = None
    similarity_score: float


class SimilarResponse(BaseModel):
    """Response payload for similar-issue retrieval."""
    results: list[SimilarIssueItem]
    retrieval_method: str = "tfidf_cosine"
    indexed_issue_count: int
