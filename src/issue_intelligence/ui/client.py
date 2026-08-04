"""HTTP client for the GitHub Issue Intelligence FastAPI backend.

Communicates exclusively via the REST API. Does not load model artifacts,
datasets, or retrieval indexes directly.

Configuration
-------------
Set ISSUE_API_URL to override the default backend address:
    export ISSUE_API_URL=http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 10.0  # seconds


def get_api_url() -> str:
    """Return the backend URL from the environment or the default."""
    return os.environ.get("ISSUE_API_URL", DEFAULT_API_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Result dataclasses (mirror the API response shapes)
# ---------------------------------------------------------------------------


@dataclass
class HealthResult:
    reachable: bool
    status: str = "unknown"
    model_loaded: bool = False
    retrieval_loaded: bool = False
    error: Optional[str] = None


@dataclass
class PredictResult:
    ok: bool
    predicted_label: str = ""
    model_name: str = ""
    decision_scores: dict[str, float] = field(default_factory=dict)
    decision_margin: float = 0.0
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class SimilarIssue:
    issue_number: int
    title: str
    target_label: str
    created_at: str
    url: str
    similarity_score: float


@dataclass
class SimilarResult:
    ok: bool
    results: list[SimilarIssue] = field(default_factory=list)
    retrieval_method: str = ""
    indexed_issue_count: int = 0
    error: Optional[str] = None
    status_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Client functions
# ---------------------------------------------------------------------------


def get_health(
    api_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> HealthResult:
    """Call GET /health and return a structured result.

    Never raises — connection failures are captured in HealthResult.error.
    """
    base = api_url or get_api_url()
    try:
        resp = httpx.get(f"{base}/health", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return HealthResult(
            reachable=True,
            status=data.get("status", "unknown"),
            model_loaded=bool(data.get("model_loaded", False)),
            retrieval_loaded=bool(data.get("retrieval_loaded", False)),
        )
    except httpx.ConnectError:
        return HealthResult(
            reachable=False,
            error="Cannot connect to the API backend.",
        )
    except httpx.TimeoutException:
        return HealthResult(
            reachable=False,
            error="Request timed out while contacting the backend.",
        )
    except Exception as exc:  # noqa: BLE001
        return HealthResult(
            reachable=False,
            error=f"Unexpected error: {type(exc).__name__}",
        )


def predict(
    title: Optional[str],
    body: Optional[str],
    api_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> PredictResult:
    """Call POST /predict and return a structured result.

    Never raises — HTTP and connection errors are captured in PredictResult.
    Decision scores are preserved exactly as returned by the API — no
    softmax, no normalisation, no conversion to probabilities.
    """
    base = api_url or get_api_url()
    payload: dict = {}
    if title is not None:
        payload["title"] = title
    if body is not None:
        payload["body"] = body

    try:
        resp = httpx.post(f"{base}/predict", json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return PredictResult(
                ok=True,
                predicted_label=data.get("predicted_label", ""),
                model_name=data.get("model_name", ""),
                decision_scores=dict(data.get("decision_scores", {})),
                decision_margin=float(data.get("decision_margin", 0.0)),
            )
        # Non-200 response
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        return PredictResult(
            ok=False,
            error=str(detail),
            status_code=resp.status_code,
        )
    except httpx.ConnectError:
        return PredictResult(ok=False, error="Cannot connect to the API backend.")
    except httpx.TimeoutException:
        return PredictResult(ok=False, error="Prediction request timed out.")
    except Exception as exc:  # noqa: BLE001
        return PredictResult(ok=False, error=f"Unexpected error: {type(exc).__name__}")


def get_similar(
    title: Optional[str],
    body: Optional[str],
    top_k: int = 5,
    label_filter: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> SimilarResult:
    """Call POST /similar and return a structured result.

    Never raises. Sort order from the backend is preserved exactly.
    Similarity scores are raw TF-IDF cosine similarity values — not
    probabilities, not duplicate-detection scores.
    """
    base = api_url or get_api_url()
    payload: dict = {"top_k": top_k}
    if title is not None:
        payload["title"] = title
    if body is not None:
        payload["body"] = body
    # Only include label_filter when a specific class is chosen
    if label_filter and label_filter.lower() != "all":
        payload["label_filter"] = label_filter

    try:
        resp = httpx.post(f"{base}/similar", json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            issues = [
                SimilarIssue(
                    issue_number=item["issue_number"],
                    title=item.get("title", ""),
                    target_label=item.get("target_label", ""),
                    created_at=item.get("created_at", ""),
                    url=item.get("url", ""),
                    similarity_score=float(item.get("similarity_score", 0.0)),
                )
                for item in data.get("results", [])
            ]
            return SimilarResult(
                ok=True,
                results=issues,
                retrieval_method=data.get("retrieval_method", "tfidf_cosine"),
                indexed_issue_count=int(data.get("indexed_issue_count", 0)),
            )
        # HTTP error
        if resp.status_code == 503:
            return SimilarResult(
                ok=False,
                error="Retrieval service is unavailable (HTTP 503).",
                status_code=503,
            )
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        return SimilarResult(
            ok=False,
            error=str(detail),
            status_code=resp.status_code,
        )
    except httpx.ConnectError:
        return SimilarResult(ok=False, error="Cannot connect to the API backend.")
    except httpx.TimeoutException:
        return SimilarResult(ok=False, error="Similar-issue request timed out.")
    except Exception as exc:  # noqa: BLE001
        return SimilarResult(ok=False, error=f"Unexpected error: {type(exc).__name__}")
