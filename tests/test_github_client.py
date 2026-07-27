"""
tests/test_github_client.py
============================
Unit tests for issue_intelligence.data.github_client.

All tests use fake HTTP responses — no real network requests are made.

We use a lightweight FakeResponse class and monkeypatch to replace
requests.Session.get.  No large mocking framework is needed.

Run with:
    pytest tests/test_github_client.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from issue_intelligence.data.github_client import (
    GitHubAPIError,
    GitHubIssueCollector,
    MissingTokenError,
    _extract_issue_fields,
    _parse_next_page_url,
)

# ---------------------------------------------------------------------------
# Helpers — fake API response data
# ---------------------------------------------------------------------------

def _make_issue(number: int, labels: list[str] | None = None) -> dict[str, Any]:
    """Return a minimal dict that looks like a GitHub API issue object."""
    return {
        "id": 1000 + number,
        "number": number,
        "title": f"Issue #{number}",
        "body": f"Body of issue {number}",
        "state": "open",
        "labels": [{"name": lb} for lb in (labels or [])],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "closed_at": None,
        "comments": 0,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "user": {"type": "User"},
    }


def _make_pr(number: int) -> dict[str, Any]:
    """Return a minimal dict that looks like a GitHub API pull request object."""
    pr = _make_issue(number)
    # The GitHub Issues API marks PRs with this key
    pr["pull_request"] = {"url": f"https://api.github.com/repos/owner/repo/pulls/{number}"}
    pr["html_url"] = f"https://github.com/owner/repo/pull/{number}"
    return pr


class FakeResponse:
    """
    Minimal stand-in for requests.Response.
    Supports .ok, .status_code, .text, .json(), .headers.
    """

    def __init__(
        self,
        data: list[dict],
        status_code: int = 200,
        headers: dict | None = None,
    ) -> None:
        self._data = data
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(data)
        self.headers = headers or {}
        self.url = "https://api.github.com/repos/owner/repo/issues"

    def json(self) -> list[dict]:
        return self._data


# ---------------------------------------------------------------------------
# Tests — _parse_next_page_url
# ---------------------------------------------------------------------------

class TestParseNextPageUrl:
    def test_returns_next_url_when_present(self) -> None:
        base = "https://api.github.com/repos/o/r/issues"
        link = (
            f'<{base}?page=2>; rel="next", '
            f'<{base}?page=10>; rel="last"'
        )
        result = _parse_next_page_url(link)
        assert result == f"{base}?page=2"

    def test_returns_none_when_no_next(self) -> None:
        link = '<https://api.github.com/repos/o/r/issues?page=10>; rel="last"'
        assert _parse_next_page_url(link) is None

    def test_returns_none_for_empty_header(self) -> None:
        assert _parse_next_page_url(None) is None
        assert _parse_next_page_url("") is None


# ---------------------------------------------------------------------------
# Tests — _extract_issue_fields
# ---------------------------------------------------------------------------

class TestExtractIssueFields:
    def test_labels_simplified_to_name_list(self) -> None:
        raw = _make_issue(1, labels=["bug", "needs-triage"])
        result = _extract_issue_fields(raw)
        assert result["labels"] == ["bug", "needs-triage"]

    def test_user_type_extracted(self) -> None:
        raw = _make_issue(1)
        result = _extract_issue_fields(raw)
        assert result["user_type"] == "User"

    def test_user_type_none_when_missing(self) -> None:
        raw = _make_issue(1)
        raw["user"] = None
        result = _extract_issue_fields(raw)
        assert result["user_type"] is None

    def test_pull_request_field_not_in_output(self) -> None:
        """Even if called on a PR dict, the pull_request key is not copied."""
        raw = _make_pr(99)
        result = _extract_issue_fields(raw)
        assert "pull_request" not in result


# ---------------------------------------------------------------------------
# Tests — GitHubIssueCollector
# ---------------------------------------------------------------------------

class TestGitHubIssueCollector:

    # ------------------------------------------------------------------
    # 1. Pull requests are excluded
    # ------------------------------------------------------------------
    def test_pull_requests_excluded(self, tmp_path: Path) -> None:
        """Items with a 'pull_request' field must not appear in the output."""
        page = [_make_issue(1), _make_pr(2), _make_issue(3), _make_pr(4)]

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),  # empty second page → stop
                ]
                collector = GitHubIssueCollector()
                issues, metadata = collector.collect(
                    owner="owner", repo="repo", max_issues=100
                )

        # Only real issues (numbers 1 and 3) should appear
        numbers = [i["number"] for i in issues]
        assert numbers == [1, 3]
        assert metadata["pull_requests_excluded"] == 2

    # ------------------------------------------------------------------
    # 2. Regular issues are retained
    # ------------------------------------------------------------------
    def test_regular_issues_retained(self) -> None:
        """All non-PR items must be present in the output."""
        page = [_make_issue(i) for i in range(1, 6)]  # 5 issues, no PRs

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                issues, _ = collector.collect(
                    owner="owner", repo="repo", max_issues=100
                )

        assert len(issues) == 5
        assert all("pull_request" not in i for i in issues)

    # ------------------------------------------------------------------
    # 3. Pagination stops after max_issues
    # ------------------------------------------------------------------
    def test_pagination_stops_at_max(self) -> None:
        """Collector must stop once max_issues regular issues are collected."""
        page1 = [_make_issue(i) for i in range(1, 101)]    # 100 issues
        page2 = [_make_issue(i) for i in range(101, 201)]  # another 100

        # The Link header tells the client there is a next page
        next_link = '<https://api.github.com/repos/o/r/issues?page=2>; rel="next"'

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page1, headers={"Link": next_link}),
                    FakeResponse(page2),
                ]
                collector = GitHubIssueCollector()
                issues, metadata = collector.collect(
                    owner="owner", repo="repo", max_issues=150
                )

        # Should stop at exactly 150, not 200
        assert len(issues) == 150
        assert metadata["regular_issues_saved"] == 150

    # ------------------------------------------------------------------
    # 4. Missing token raises MissingTokenError
    # ------------------------------------------------------------------
    def test_missing_token_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing token must raise MissingTokenError, not expose any secret."""
        # Remove GITHUB_TOKEN from the environment entirely
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        # Also ensure .env is not accidentally loaded during the test
        with patch("issue_intelligence.data.github_client.load_dotenv"):
            collector = GitHubIssueCollector(require_token=True)
            with pytest.raises(MissingTokenError) as exc_info:
                collector.collect(owner="owner", repo="repo")

        # The error message must not contain any token value
        assert "ghp_" not in str(exc_info.value)
        assert "GITHUB_TOKEN" in str(exc_info.value)

    # ------------------------------------------------------------------
    # 5. HTTP errors produce a clear exception
    # ------------------------------------------------------------------
    def test_http_error_raises_github_api_error(self) -> None:
        """A non-200 HTTP status must raise GitHubAPIError."""
        error_response = FakeResponse(
            data={"message": "Not Found"},
            status_code=404,
        )
        # FakeResponse.text for a dict is fine here
        error_response.text = '{"message": "Not Found"}'

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get", return_value=error_response):
                collector = GitHubIssueCollector()
                with pytest.raises(GitHubAPIError) as exc_info:
                    collector.collect(owner="owner", repo="repo")

        assert "404" in str(exc_info.value)

    # ------------------------------------------------------------------
    # 6. Saved JSON contains the expected issue records
    # ------------------------------------------------------------------
    def test_saved_json_contains_expected_records(self, tmp_path: Path) -> None:
        """The output JSON file must contain exactly the regular issue records."""
        page = [_make_issue(10, labels=["bug"]), _make_pr(11), _make_issue(12)]
        output_file = tmp_path / "issues.json"

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                collector.collect(
                    owner="owner",
                    repo="repo",
                    max_issues=100,
                    output_path=output_file,
                )

        assert output_file.exists(), "Output JSON file was not created"
        saved = json.loads(output_file.read_text(encoding="utf-8"))

        assert isinstance(saved, list)
        assert len(saved) == 2  # PR #11 excluded

        numbers = [item["number"] for item in saved]
        assert 10 in numbers
        assert 12 in numbers
        assert 11 not in numbers

        # No entry should contain a pull_request field
        for item in saved:
            assert "pull_request" not in item

    # ------------------------------------------------------------------
    # 7. Metadata JSON is saved alongside the issues file
    # ------------------------------------------------------------------
    def test_metadata_file_saved(self, tmp_path: Path) -> None:
        """A _metadata.json file must be created next to the issues file."""
        page = [_make_issue(1)]
        output_file = tmp_path / "issues.json"

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                _, metadata = collector.collect(
                    owner="owner", repo="repo", max_issues=100, output_path=output_file
                )

        meta_file = tmp_path / "issues_metadata.json"
        assert meta_file.exists(), "Metadata JSON file was not created"
        saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))

        # Token must never appear in metadata
        assert "token" not in str(saved_meta).lower()
        assert "GITHUB_TOKEN" not in str(saved_meta)
        # authenticated is a boolean
        assert isinstance(saved_meta["authenticated"], bool)

    # ------------------------------------------------------------------
    # Stage 1C — new tests
    # ------------------------------------------------------------------

    # 8. Sort and direction parameters are passed to the API
    def test_sort_and_direction_passed_to_api(self) -> None:
        """Collector must pass sort and direction to the GitHub API."""
        page = [_make_issue(1)]

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                _, metadata = collector.collect(
                    owner="owner",
                    repo="repo",
                    sort="created",
                    direction="asc",
                    max_issues=10,
                )

        # Check that the first call included sort and direction params
        first_call_kwargs = mock_get.call_args_list[0]
        params = first_call_kwargs[1].get("params") or first_call_kwargs[0][1]
        assert params.get("sort") == "created"
        assert params.get("direction") == "asc"

    # 9. Metadata contains sort and direction fields
    def test_metadata_contains_sort_and_direction(self) -> None:
        """Metadata must record the sort and direction used during collection."""
        page = [_make_issue(1)]

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                _, metadata = collector.collect(
                    owner="owner",
                    repo="repo",
                    sort="created",
                    direction="asc",
                    max_issues=10,
                )

        assert metadata["sort"] == "created"
        assert metadata["direction"] == "asc"

    # 10. Metadata contains date coverage fields
    def test_metadata_contains_date_coverage(self) -> None:
        """Metadata must record earliest and latest creation dates."""
        issue_a = _make_issue(1)
        issue_a["created_at"] = "2015-03-01T00:00:00Z"
        issue_b = _make_issue(2)
        issue_b["created_at"] = "2020-06-15T00:00:00Z"
        page = [issue_a, issue_b]

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                _, metadata = collector.collect(
                    owner="owner", repo="repo", max_issues=100
                )

        assert metadata["earliest_created_at"] == "2015-03-01T00:00:00Z"
        assert metadata["latest_created_at"] == "2020-06-15T00:00:00Z"

    # 11. Deduplication skips issues already seen by ID
    def test_deduplication_skips_duplicate_ids(self) -> None:
        """The same issue ID appearing twice must only be saved once."""
        dup_issue = _make_issue(42)  # id = 1042
        page1 = [dup_issue, _make_issue(1)]
        page2 = [dup_issue, _make_issue(2)]  # dup_issue appears again

        next_link = '<https://api.github.com/repos/o/r/issues?page=2>; rel="next"'

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page1, headers={"Link": next_link}),
                    FakeResponse(page2),
                ]
                collector = GitHubIssueCollector()
                issues, metadata = collector.collect(
                    owner="owner", repo="repo", max_issues=100
                )

        ids = [i["id"] for i in issues]
        # The duplicate issue ID (1042) should appear exactly once
        assert ids.count(dup_issue["id"]) == 1
        # Total unique issues = 3 (42, 1, 2)
        assert len(issues) == 3
        assert metadata["duplicates_skipped"] == 1

    # 12. metadata contains duplicates_skipped key
    def test_metadata_contains_duplicates_skipped(self) -> None:
        """Metadata must always include a 'duplicates_skipped' key."""
        page = [_make_issue(1)]

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                _, metadata = collector.collect(
                    owner="owner", repo="repo", max_issues=10
                )

        assert "duplicates_skipped" in metadata
        assert metadata["duplicates_skipped"] == 0

    # 13. No pull requests in saved output (Stage 1C recheck)
    def test_no_prs_in_history_output(self, tmp_path: Path) -> None:
        """All saved issues must be free of the pull_request field."""
        page = [_make_issue(i) for i in range(1, 5)] + [_make_pr(99)]
        output_file = tmp_path / "history.json"

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [
                    FakeResponse(page),
                    FakeResponse([]),
                ]
                collector = GitHubIssueCollector()
                collector.collect(
                    owner="owner",
                    repo="repo",
                    sort="created",
                    direction="asc",
                    max_issues=100,
                    output_path=output_file,
                )

        saved = json.loads(output_file.read_text(encoding="utf-8"))
        assert all("pull_request" not in i for i in saved)
        assert len(saved) == 4  # 4 issues, 1 PR excluded

    # 14. Output path does not overwrite a different file
    def test_different_output_paths_are_independent(
        self, tmp_path: Path
    ) -> None:
        """Collecting to two different paths must not overwrite each other."""
        page_a = [_make_issue(1)]
        page_b = [_make_issue(999)]
        path_a = tmp_path / "sample.json"
        path_b = tmp_path / "history.json"

        with patch.dict("os.environ", {"GITHUB_TOKEN": "fake-token"}):
            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [FakeResponse(page_a), FakeResponse([])]
                GitHubIssueCollector().collect(
                    owner="o", repo="r", max_issues=10, output_path=path_a
                )

            with patch("requests.Session.get") as mock_get:
                mock_get.side_effect = [FakeResponse(page_b), FakeResponse([])]
                GitHubIssueCollector().collect(
                    owner="o", repo="r", max_issues=10, output_path=path_b
                )

        data_a = json.loads(path_a.read_text(encoding="utf-8"))
        data_b = json.loads(path_b.read_text(encoding="utf-8"))
        assert data_a[0]["number"] == 1
        assert data_b[0]["number"] == 999
        # The original sample must be untouched
        assert data_a[0]["number"] != data_b[0]["number"]
