"""
issue_intelligence.data.github_client
======================================
Reusable GitHub Issues API client.

Fetches regular issues (excluding pull requests) from any public
GitHub repository, handles pagination, rate limiting, deduplication,
and saves results as UTF-8 JSON.

Usage example
-------------
from issue_intelligence.data.github_client import GitHubIssueCollector

collector = GitHubIssueCollector()
issues, metadata = collector.collect(
    owner="scikit-learn",
    repo="scikit-learn",
    state="all",
    sort="created",
    direction="asc",        # oldest-first for historical coverage
    max_issues=5000,
    output_path="data/raw/scikit-learn_issues_history.json",
)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Module-level logger — never logs the token value
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Raised when the GitHub API returns an unexpected HTTP error."""


class MissingTokenError(Exception):
    """Raised when GITHUB_TOKEN is absent from the environment."""


# Fields we keep from each raw issue object.
# We preserve them as-is — no preprocessing at collection time.
_FIELDS_TO_KEEP = [
    "id",
    "number",
    "title",
    "body",
    "state",
    "labels",
    "created_at",
    "updated_at",
    "closed_at",
    "comments",
    "html_url",
]


def _load_token() -> str | None:
    """
    Load GITHUB_TOKEN from the environment (or .env file).

    Returns the token string, or None if not set.
    The token value is never logged.
    """
    load_dotenv()  # reads .env if present; does nothing if absent
    return os.environ.get("GITHUB_TOKEN")


def _build_session(token: str | None) -> tuple[requests.Session, bool]:
    """
    Build a requests.Session with appropriate headers.

    Returns (session, authenticated) where `authenticated` is a boolean
    that can be stored in metadata without revealing the token.
    """
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    if token:
        # The Authorization header value is never stored or logged.
        session.headers["Authorization"] = f"Bearer {token}"
        return session, True
    return session, False


def _extract_issue_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Extract only the fields we need from a raw API issue object.

    Labels are simplified to a list of label name strings.
    The user's account type (e.g. 'User', 'Bot') is preserved
    without storing any personal information.
    """
    issue: dict[str, Any] = {}
    for field in _FIELDS_TO_KEEP:
        issue[field] = raw.get(field)

    # Simplify labels from [{id, name, color, ...}] to [name, ...]
    raw_labels = raw.get("labels") or []
    issue["labels"] = [lb["name"] for lb in raw_labels if "name" in lb]

    # Store only the user's account type (e.g. 'User', 'Bot')
    user = raw.get("user") or {}
    issue["user_type"] = user.get("type")

    return issue


def _parse_next_page_url(link_header: str | None) -> str | None:
    """
    Parse the GitHub Link header to find the URL for the next page.

    GitHub sends:  Link: <url>; rel="next", <url>; rel="last"
    Returns the next-page URL, or None if there is no next page.
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            # Extract the URL between angle brackets
            url_part = part.split(";")[0].strip()
            if url_part.startswith("<") and url_part.endswith(">"):
                return url_part[1:-1]
    return None


class GitHubIssueCollector:
    """
    Collects GitHub issues via the REST API.

    Parameters
    ----------
    timeout : int
        Request timeout in seconds (default: 30).
    require_token : bool
        If True (default), raise MissingTokenError when the token is absent.
        Set to False only in tests that deliberately omit the token.
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, timeout: int = 30, require_token: bool = True) -> None:
        self.timeout = timeout
        self.require_token = require_token

    def collect(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        sort: str = "created",
        direction: str = "desc",
        max_issues: int = 500,
        output_path: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Collect regular GitHub issues (pull requests excluded).

        Parameters
        ----------
        owner : str
            GitHub repository owner (e.g. "scikit-learn").
        repo : str
            Repository name (e.g. "scikit-learn").
        state : str
            "open", "closed", or "all".
        sort : str
            Sort field: "created", "updated", or "comments".
        direction : str
            "asc" (oldest first) or "desc" (newest first, default).
            Use "asc" for maximum historical coverage.
        max_issues : int
            Maximum number of *regular issues* to collect (not API items).
        output_path : str or Path, optional
            If provided, save issues JSON and metadata JSON here.

        Returns
        -------
        issues : list of dict
            Collected regular issue records (deduplicated by GitHub issue ID).
        metadata : dict
            Collection statistics and provenance (no token value).
        """
        token = _load_token()
        if self.require_token and not token:
            raise MissingTokenError(
                "GITHUB_TOKEN is not set. "
                "Copy .env.example to .env and add your token."
            )

        session, authenticated = _build_session(token)

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
        params: dict[str, Any] = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": 100,
            "page": 1,
        }

        issues: list[dict[str, Any]] = []
        seen_ids: set[int] = set()  # for deduplication

        api_items_inspected = 0
        pull_requests_excluded = 0
        duplicates_skipped = 0
        remaining_rate_limit: int | None = None
        # tracked separately; params are cleared after Link-header pagination
        page_number = 1

        logger.info(
            "Starting collection: %s/%s  state=%s  sort=%s  direction=%s  max=%d",
            owner, repo, state, sort, direction, max_issues,
        )

        while len(issues) < max_issues:
            logger.debug("Fetching page %d …", page_number)

            try:
                response = session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.Timeout:
                raise GitHubAPIError(
                    f"Request timed out after {self.timeout}s. "
                    "Try again or increase the timeout."
                )
            except requests.exceptions.ConnectionError as exc:
                raise GitHubAPIError(f"Network error: {exc}") from exc

            # Surface rate-limit information before raising on errors
            remaining_rate_limit = self._parse_rate_limit(response)

            if response.status_code == 403 and "rate limit" in response.text.lower():
                reset_ts = response.headers.get("X-RateLimit-Reset", "unknown")
                raise GitHubAPIError(
                    f"GitHub API rate limit exceeded. "
                    f"Resets at Unix timestamp: {reset_ts}. "
                    f"Remaining: {remaining_rate_limit}"
                )

            if not response.ok:
                raise GitHubAPIError(
                    f"GitHub API returned HTTP {response.status_code} "
                    f"for {response.url}: {response.text[:200]}"
                )

            page_items: list[dict[str, Any]] = response.json()
            if not page_items:
                # No more items on this page — we've exhausted the API
                break

            for raw_item in page_items:
                api_items_inspected += 1

                # The GitHub Issues API returns PRs mixed with issues.
                # Any item with a "pull_request" key is a PR — exclude it.
                if "pull_request" in raw_item:
                    pull_requests_excluded += 1
                    continue

                # Deduplicate by GitHub issue ID (safety measure for
                # edge cases at pagination boundaries)
                issue_id = raw_item.get("id")
                if issue_id is not None and issue_id in seen_ids:
                    duplicates_skipped += 1
                    continue
                if issue_id is not None:
                    seen_ids.add(issue_id)

                issues.append(_extract_issue_fields(raw_item))

                if len(issues) >= max_issues:
                    break  # reached the requested maximum

            # Check for a next page via the Link header
            next_url = _parse_next_page_url(response.headers.get("Link"))
            if next_url is None:
                break  # no more pages

            # Use the full next URL directly (it already has all params)
            url = next_url
            params = {}  # params are encoded in the next URL
            page_number += 1

        logger.info(
            "Collection complete: %d API items inspected, "
            "%d PRs excluded, %d duplicates skipped, %d issues saved.",
            api_items_inspected,
            pull_requests_excluded,
            duplicates_skipped,
            len(issues),
        )

        # Compute date coverage from the collected issues
        dates = [
            i["created_at"] for i in issues if i.get("created_at")
        ]
        earliest_created_at = min(dates) if dates else None
        latest_created_at = max(dates) if dates else None

        metadata = self._build_metadata(
            owner=owner,
            repo=repo,
            state=state,
            sort=sort,
            direction=direction,
            max_issues=max_issues,
            api_items_inspected=api_items_inspected,
            pull_requests_excluded=pull_requests_excluded,
            duplicates_skipped=duplicates_skipped,
            issues_saved=len(issues),
            authenticated=authenticated,
            remaining_rate_limit=remaining_rate_limit,
            earliest_created_at=earliest_created_at,
            latest_created_at=latest_created_at,
        )

        if output_path is not None:
            self._save(issues, metadata, Path(output_path))

        return issues, metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rate_limit(response: requests.Response) -> int | None:
        """Extract the remaining rate-limit count from response headers."""
        raw = response.headers.get("X-RateLimit-Remaining")
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
        return None

    @staticmethod
    def _build_metadata(
        *,
        owner: str,
        repo: str,
        state: str,
        sort: str,
        direction: str,
        max_issues: int,
        api_items_inspected: int,
        pull_requests_excluded: int,
        duplicates_skipped: int,
        issues_saved: int,
        authenticated: bool,
        remaining_rate_limit: int | None,
        earliest_created_at: str | None,
        latest_created_at: str | None,
    ) -> dict[str, Any]:
        """Build the metadata dictionary. The token is never included."""
        return {
            "repository": f"{owner}/{repo}",
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "state_filter": state,
            "sort": sort,
            "direction": direction,
            "requested_maximum": max_issues,
            "api_items_inspected": api_items_inspected,
            "pull_requests_excluded": pull_requests_excluded,
            "duplicates_skipped": duplicates_skipped,
            "regular_issues_saved": issues_saved,
            "authenticated": authenticated,  # boolean only — no token
            "remaining_rate_limit": remaining_rate_limit,
            "earliest_created_at": earliest_created_at,
            "latest_created_at": latest_created_at,
        }

    @staticmethod
    def _save(
        issues: list[dict[str, Any]],
        metadata: dict[str, Any],
        output_path: Path,
    ) -> None:
        """Save issues and metadata as UTF-8 JSON files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the issues
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(issues, fh, ensure_ascii=False, indent=2)
        logger.info("Issues saved -> %s", output_path)

        # Save the metadata alongside the issues file
        meta_path = output_path.with_name(
            output_path.stem + "_metadata" + output_path.suffix
        )
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
        logger.info("Metadata saved -> %s", meta_path)
