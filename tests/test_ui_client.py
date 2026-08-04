"""Tests for the UI API client (issue_intelligence.ui.client).

All tests use mocked HTTP responses via pytest-monkeypatch or unittest.mock.
No real backend, model artifacts, datasets, or retrieval indexes are used.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from issue_intelligence.ui.client import (
    get_health,
    get_similar,
    predict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PREDICT_PAYLOAD = {
    "predicted_label": "Bug",
    "model_name": "LinearSVC",
    "decision_scores": {"Bug": 1.23, "Documentation": -0.45, "Enhancement": -0.78},
    "decision_margin": 1.68,
}

_SIMILAR_PAYLOAD = {
    "results": [
        {
            "issue_number": 14613,
            "title": "EllipticEnvelope does not work with a sparse matrix",
            "target_label": "Documentation",
            "created_at": "2019-08-09T13:17:27Z",
            "url": "https://github.com/scikit-learn/scikit-learn/issues/14613",
            "similarity_score": 0.214,
        },
        {
            "issue_number": 1324,
            "title": "MinMaxScaler does not support sparse input.",
            "target_label": "Enhancement",
            "created_at": "2012-11-04T09:12:18Z",
            "url": "https://github.com/scikit-learn/scikit-learn/issues/1324",
            "similarity_score": 0.194,
        },
    ],
    "retrieval_method": "tfidf_cosine",
    "indexed_issue_count": 4854,
}


def _mock_response(status_code: int, json_data: dict) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


# ---------------------------------------------------------------------------
# get_health tests
# ---------------------------------------------------------------------------


class TestGetHealth:
    def test_health_success(self):
        resp = _mock_response(
            200,
            {"status": "ok", "model_loaded": True, "retrieval_loaded": True},
        )
        with patch("httpx.get", return_value=resp):
            result = get_health(api_url="http://test")
        assert result.reachable is True
        assert result.status == "ok"
        assert result.model_loaded is True
        assert result.retrieval_loaded is True
        assert result.error is None

    def test_health_model_not_loaded(self):
        resp = _mock_response(
            200,
            {"status": "not_ready", "model_loaded": False, "retrieval_loaded": False},
        )
        with patch("httpx.get", return_value=resp):
            result = get_health(api_url="http://test")
        assert result.reachable is True
        assert result.model_loaded is False
        assert result.retrieval_loaded is False

    def test_backend_connection_failure(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = get_health(api_url="http://test")
        assert result.reachable is False
        assert result.error is not None
        assert "connect" in result.error.lower()

    def test_request_timeout(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            result = get_health(api_url="http://test")
        assert result.reachable is False
        assert result.error is not None
        assert "timed out" in result.error.lower()

    def test_unexpected_exception(self):
        with patch("httpx.get", side_effect=RuntimeError("boom")):
            result = get_health(api_url="http://test")
        assert result.reachable is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# predict tests
# ---------------------------------------------------------------------------


class TestPredict:
    def test_prediction_success(self):
        resp = _mock_response(200, _PREDICT_PAYLOAD)
        with patch("httpx.post", return_value=resp):
            result = predict(title="sparse matrix error", body=None, api_url="http://test")
        assert result.ok is True
        assert result.predicted_label == "Bug"
        assert result.model_name == "LinearSVC"
        assert result.decision_margin == pytest.approx(1.68)

    def test_decision_scores_preserved_exactly(self):
        resp = _mock_response(200, _PREDICT_PAYLOAD)
        with patch("httpx.post", return_value=resp):
            result = predict(title="test", body=None, api_url="http://test")
        assert result.decision_scores["Bug"] == pytest.approx(1.23)
        assert result.decision_scores["Documentation"] == pytest.approx(-0.45)
        assert result.decision_scores["Enhancement"] == pytest.approx(-0.78)

    def test_negative_decision_scores_preserved(self):
        """Negative scores must not be clipped or converted."""
        payload = dict(_PREDICT_PAYLOAD)
        payload["decision_scores"] = {
            "Bug": 2.0, "Documentation": -3.5, "Enhancement": -1.0
        }
        resp = _mock_response(200, payload)
        with patch("httpx.post", return_value=resp):
            result = predict(title="x", body=None, api_url="http://test")
        assert result.decision_scores["Documentation"] == pytest.approx(-3.5)

    def test_no_conversion_to_probabilities(self):
        """Scores must sum to something other than 1.0 — no softmax applied."""
        resp = _mock_response(200, _PREDICT_PAYLOAD)
        with patch("httpx.post", return_value=resp):
            result = predict(title="x", body=None, api_url="http://test")
        total = sum(result.decision_scores.values())
        # Raw scores sum to 1.23 - 0.45 - 0.78 = 0.0, but certainly not guaranteed 1.0
        assert abs(total - 1.0) > 0.01 or len(result.decision_scores) != 3  # not probs

    def test_prediction_http_error_503(self):
        resp = _mock_response(503, {"detail": "Model is not loaded"})
        with patch("httpx.post", return_value=resp):
            result = predict(title="x", body=None, api_url="http://test")
        assert result.ok is False
        assert result.status_code == 503
        assert result.error is not None

    def test_prediction_http_error_422(self):
        resp = _mock_response(422, {"detail": "title and body cannot both be blank"})
        with patch("httpx.post", return_value=resp):
            result = predict(title=None, body=None, api_url="http://test")
        assert result.ok is False
        assert result.status_code == 422

    def test_connection_failure(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = predict(title="x", body=None, api_url="http://test")
        assert result.ok is False
        assert "connect" in (result.error or "").lower()

    def test_request_timeout(self):
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            result = predict(title="x", body=None, api_url="http://test")
        assert result.ok is False
        assert "timed out" in (result.error or "").lower()

    def test_title_only_input(self):
        resp = _mock_response(200, _PREDICT_PAYLOAD)
        with patch("httpx.post", return_value=resp) as mock_post:
            predict(title="sparse bug", body=None, api_url="http://test")
        sent = mock_post.call_args
        assert sent.kwargs["json"].get("title") == "sparse bug"
        assert "body" not in sent.kwargs["json"]

    def test_body_only_input(self):
        resp = _mock_response(200, _PREDICT_PAYLOAD)
        with patch("httpx.post", return_value=resp) as mock_post:
            predict(title=None, body="the body text", api_url="http://test")
        sent = mock_post.call_args
        assert sent.kwargs["json"].get("body") == "the body text"
        assert "title" not in sent.kwargs["json"]

    def test_malformed_response_missing_fields(self):
        """Partial response must not crash — defaults apply."""
        resp = _mock_response(200, {"predicted_label": "Bug"})
        with patch("httpx.post", return_value=resp):
            result = predict(title="x", body=None, api_url="http://test")
        assert result.ok is True
        assert result.predicted_label == "Bug"
        assert result.decision_scores == {}


# ---------------------------------------------------------------------------
# get_similar tests
# ---------------------------------------------------------------------------


class TestGetSimilar:
    def test_retrieval_success(self):
        resp = _mock_response(200, _SIMILAR_PAYLOAD)
        with patch("httpx.post", return_value=resp):
            result = get_similar(title="sparse error", body=None, api_url="http://test")
        assert result.ok is True
        assert len(result.results) == 2
        assert result.retrieval_method == "tfidf_cosine"
        assert result.indexed_issue_count == 4854

    def test_issue_urls_and_numbers_preserved(self):
        resp = _mock_response(200, _SIMILAR_PAYLOAD)
        with patch("httpx.post", return_value=resp):
            result = get_similar(title="sparse", body=None, api_url="http://test")
        first = result.results[0]
        assert first.issue_number == 14613
        assert "14613" in first.url

    def test_sort_order_preserved(self):
        """Backend sort order must not be changed client-side."""
        resp = _mock_response(200, _SIMILAR_PAYLOAD)
        with patch("httpx.post", return_value=resp):
            result = get_similar(title="x", body=None, api_url="http://test")
        scores = [r.similarity_score for r in result.results]
        assert scores == sorted(scores, reverse=True)  # backend already sorted desc

    def test_retrieval_503(self):
        resp = _mock_response(503, {"detail": "Retrieval artifact is not loaded"})
        with patch("httpx.post", return_value=resp):
            result = get_similar(title="x", body=None, api_url="http://test")
        assert result.ok is False
        assert result.status_code == 503

    def test_empty_results_handled(self):
        payload = dict(_SIMILAR_PAYLOAD)
        payload["results"] = []
        resp = _mock_response(200, payload)
        with patch("httpx.post", return_value=resp):
            result = get_similar(title="xyzzy nonsense", body=None, api_url="http://test")
        assert result.ok is True
        assert result.results == []

    def test_label_filter_omitted_for_all(self):
        resp = _mock_response(200, _SIMILAR_PAYLOAD)
        with patch("httpx.post", return_value=resp) as mock_post:
            get_similar(title="x", body=None, label_filter="All", api_url="http://test")
        sent = mock_post.call_args.kwargs["json"]
        assert "label_filter" not in sent

    def test_label_filter_included_for_specific_class(self):
        resp = _mock_response(200, _SIMILAR_PAYLOAD)
        with patch("httpx.post", return_value=resp) as mock_post:
            get_similar(title="x", body=None, label_filter="Bug", api_url="http://test")
        sent = mock_post.call_args.kwargs["json"]
        assert sent.get("label_filter") == "Bug"

    def test_top_k_passed_correctly(self):
        resp = _mock_response(200, _SIMILAR_PAYLOAD)
        with patch("httpx.post", return_value=resp) as mock_post:
            get_similar(title="x", body=None, top_k=3, api_url="http://test")
        sent = mock_post.call_args.kwargs["json"]
        assert sent.get("top_k") == 3

    def test_connection_failure(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = get_similar(title="x", body=None, api_url="http://test")
        assert result.ok is False
        assert "connect" in (result.error or "").lower()

    def test_request_timeout(self):
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            result = get_similar(title="x", body=None, api_url="http://test")
        assert result.ok is False
        assert "timed out" in (result.error or "").lower()

    def test_malformed_response(self):
        """A response missing required result fields must not crash."""
        payload = {
            "results": [{"issue_number": 999}],  # missing most fields
            "retrieval_method": "tfidf_cosine",
            "indexed_issue_count": 4854,
        }
        resp = _mock_response(200, payload)
        with patch("httpx.post", return_value=resp):
            result = get_similar(title="x", body=None, api_url="http://test")
        assert result.ok is True
        assert result.results[0].issue_number == 999


# ---------------------------------------------------------------------------
# Independent failure tests
# ---------------------------------------------------------------------------


class TestIndependentFailure:
    def test_classification_succeeds_when_retrieval_fails(self):
        """Verify that predict() and get_similar() are fully independent calls."""
        good_predict = _mock_response(200, _PREDICT_PAYLOAD)
        bad_similar = _mock_response(503, {"detail": "not loaded"})

        with patch("httpx.post", side_effect=[good_predict, bad_similar]):
            pred = predict(title="x", body=None, api_url="http://test")
            sim = get_similar(title="x", body=None, api_url="http://test")

        assert pred.ok is True
        assert sim.ok is False

    def test_retrieval_succeeds_when_classification_fails(self):
        """Retrieval result is still valid even when classification fails."""
        bad_predict = _mock_response(503, {"detail": "not loaded"})
        good_similar = _mock_response(200, _SIMILAR_PAYLOAD)

        with patch("httpx.post", side_effect=[bad_predict, good_similar]):
            pred = predict(title="x", body=None, api_url="http://test")
            sim = get_similar(title="x", body=None, api_url="http://test")

        assert pred.ok is False
        assert sim.ok is True
        assert len(sim.results) == 2
