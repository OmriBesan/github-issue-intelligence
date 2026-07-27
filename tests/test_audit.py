"""
tests/test_audit.py
====================
Unit tests for issue_intelligence.data.audit.

All tests use small synthetic issue records — the real raw dataset
is never loaded here.

Run with:
    pytest tests/test_audit.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from issue_intelligence.data.audit import (
    PROVISIONAL_TYPE_LABELS,
    build_label_scheme_stats,
    build_provisional_subset,
    compute_label_cooccurrence,
    compute_label_distribution_by_year,
    compute_label_frequencies,
    compute_labels_per_issue,
    compute_missing_data,
    compute_text_lengths,
    compute_yearly_distribution,
    detect_leakage,
    load_issues,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_issue(
    number: int,
    title: str = "Sample issue title",
    body: str = "Sample body text.",
    labels: list[str] | None = None,
    state: str = "open",
    created_at: str = "2024-06-15T10:00:00Z",
    comments: int = 0,
) -> dict[str, Any]:
    return {
        "id": 90000 + number,
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "labels": labels or [],
        "created_at": created_at,
        "updated_at": created_at,
        "closed_at": None,
        "comments": comments,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "user_type": "User",
    }


# ---------------------------------------------------------------------------
# 1. Label frequencies are counted correctly
# ---------------------------------------------------------------------------


class TestComputeLabelFrequencies:
    def test_counts_single_labels(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug"]),
            _make_issue(2, labels=["Bug"]),
            _make_issue(3, labels=["Documentation"]),
        ]
        freq = compute_label_frequencies(issues)
        assert freq["Bug"] == 2
        assert freq["Documentation"] == 1

    def test_unlabelled_issues_add_nothing(self) -> None:
        issues = [_make_issue(1, labels=[]), _make_issue(2, labels=None)]
        freq = compute_label_frequencies(issues)
        assert len(freq) == 0

    def test_total_count_equals_sum_of_all_labels(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "Needs Triage"]),
            _make_issue(2, labels=["Documentation"]),
        ]
        freq = compute_label_frequencies(issues)
        assert sum(freq.values()) == 3


# ---------------------------------------------------------------------------
# 2. Multiple labels are handled correctly
# ---------------------------------------------------------------------------


class TestComputeLabelsPerIssue:
    def test_correct_counts_per_issue(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "Needs Triage"]),
            _make_issue(2, labels=["Documentation"]),
            _make_issue(3, labels=[]),
        ]
        counts = compute_labels_per_issue(issues)
        assert counts == [2, 1, 0]

    def test_none_labels_treated_as_zero(self) -> None:
        issues = [_make_issue(1, labels=None)]
        counts = compute_labels_per_issue(issues)
        assert counts == [0]

    def test_multi_label_percentage(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "RFC"]),
            _make_issue(2, labels=["Documentation"]),
            _make_issue(3, labels=[]),
        ]
        counts = compute_labels_per_issue(issues)
        multi = sum(1 for c in counts if c > 1)
        assert multi == 1  # one issue with 2 labels


# ---------------------------------------------------------------------------
# 3. Missing bodies are detected
# ---------------------------------------------------------------------------


class TestComputeMissingData:
    def test_none_body_counted(self) -> None:
        issues = [_make_issue(1, body=None)]  # type: ignore[arg-type]
        result = compute_missing_data(issues)
        assert result["missing_body"] == 1
        assert result["total_body_problems"] == 1

    def test_whitespace_only_body_counted_as_empty(self) -> None:
        issues = [_make_issue(1, body="   \n  ")]
        result = compute_missing_data(issues)
        assert result["empty_body"] == 1
        assert result["total_body_problems"] == 1

    def test_normal_body_not_counted(self) -> None:
        issues = [_make_issue(1, body="A perfectly fine body.")]
        result = compute_missing_data(issues)
        assert result["total_body_problems"] == 0

    def test_none_title_counted(self) -> None:
        issues = [_make_issue(1, title=None)]  # type: ignore[arg-type]
        result = compute_missing_data(issues)
        assert result["missing_title"] == 1


# ---------------------------------------------------------------------------
# 4. Pull-request fields are not expected in audit input
#    (The collector already filters them out — we just confirm they are absent)
# ---------------------------------------------------------------------------


class TestNoPullRequestsInAuditInput:
    def test_no_pull_request_key_in_synthetic_data(self) -> None:
        issues = [_make_issue(i) for i in range(5)]
        for issue in issues:
            assert "pull_request" not in issue, (
                f"Issue {issue['number']} unexpectedly has a pull_request key"
            )


# ---------------------------------------------------------------------------
# 5. Text lengths are computed correctly
# ---------------------------------------------------------------------------


class TestComputeTextLengths:
    def test_title_length(self) -> None:
        issues = [_make_issue(1, title="Hello", body="")]
        lengths = compute_text_lengths(issues)
        assert lengths["title"] == [5]

    def test_body_length(self) -> None:
        issues = [_make_issue(1, title="", body="World!")]
        lengths = compute_text_lengths(issues)
        assert lengths["body"] == [6]

    def test_combined_length(self) -> None:
        issues = [_make_issue(1, title="Hi", body="There")]
        lengths = compute_text_lengths(issues)
        assert lengths["combined"] == [7]  # 2 + 5

    def test_none_treated_as_zero(self) -> None:
        issues = [_make_issue(1, title=None, body=None)]  # type: ignore[arg-type]
        lengths = compute_text_lengths(issues)
        assert lengths["title"] == [0]
        assert lengths["body"] == [0]
        assert lengths["combined"] == [0]


# ---------------------------------------------------------------------------
# 6. Label co-occurrence is counted correctly
# ---------------------------------------------------------------------------


class TestComputeLabelCooccurrence:
    def test_pair_counted(self) -> None:
        issues = [_make_issue(1, labels=["Bug", "Needs Triage"])]
        cooc = compute_label_cooccurrence(issues)
        assert cooc[("Bug", "Needs Triage")] == 1

    def test_pair_appears_twice(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "Needs Triage"]),
            _make_issue(2, labels=["Bug", "Needs Triage"]),
        ]
        cooc = compute_label_cooccurrence(issues)
        assert cooc[("Bug", "Needs Triage")] == 2

    def test_selected_labels_filter(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "Needs Triage", "Documentation"])
        ]
        # Only track Bug and Documentation
        cooc = compute_label_cooccurrence(
            issues, selected_labels=["Bug", "Documentation"]
        )
        assert ("Bug", "Documentation") in cooc
        # Needs Triage is excluded
        assert all("Needs Triage" not in str(k) for k in cooc)

    def test_single_label_produces_no_pairs(self) -> None:
        issues = [_make_issue(1, labels=["Bug"])]
        cooc = compute_label_cooccurrence(issues)
        assert len(cooc) == 0

    def test_pairs_stored_lexicographically(self) -> None:
        issues = [_make_issue(1, labels=["Zeta", "Alpha"])]
        cooc = compute_label_cooccurrence(issues)
        # Pair must be stored as ("Alpha", "Zeta"), not ("Zeta", "Alpha")
        assert ("Alpha", "Zeta") in cooc
        assert ("Zeta", "Alpha") not in cooc


# ---------------------------------------------------------------------------
# 7. Candidate type-label overlap is detected
# ---------------------------------------------------------------------------


class TestBuildProvisionalSubsetOverlap:
    def test_multi_type_issue_detected(self) -> None:
        issues = [_make_issue(1, labels=["Bug", "Documentation"])]
        result = build_provisional_subset(issues)
        assert result["excluded_multi_type"] == 1
        assert len(result["type_label_overlap"]) == 1
        number, matched = result["type_label_overlap"][0]
        assert number == 1
        assert set(matched) == {"Bug", "Documentation"}


# ---------------------------------------------------------------------------
# 8. The provisional single-type subset is built correctly
# ---------------------------------------------------------------------------


class TestBuildProvisionalSubset:
    def test_single_type_issue_included(self) -> None:
        issues = [_make_issue(1, labels=["Bug"])]
        result = build_provisional_subset(issues)
        assert result["total_usable"] == 1
        assert result["class_counts"]["Bug"] == 1

    def test_unlabelled_issue_excluded(self) -> None:
        issues = [_make_issue(1, labels=[])]
        result = build_provisional_subset(issues)
        assert result["excluded_unlabelled"] == 1
        assert result["total_usable"] == 0

    def test_non_type_label_issue_excluded(self) -> None:
        issues = [_make_issue(1, labels=["Needs Triage"])]
        result = build_provisional_subset(issues)
        assert result["excluded_unlabelled"] == 1

    def test_mixed_batch(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug"]),                    # included
            _make_issue(2, labels=["Bug", "Documentation"]),   # multi-type
            _make_issue(3, labels=["Needs Triage"]),           # no type label
            _make_issue(4, labels=[]),                         # unlabelled
            _make_issue(5, labels=["New Feature"]),            # included
        ]
        result = build_provisional_subset(issues)
        assert result["total_usable"] == 2
        assert result["excluded_multi_type"] == 1
        assert result["excluded_unlabelled"] == 2
        assert result["class_counts"]["Bug"] == 1
        assert result["class_counts"]["New Feature"] == 1


# ---------------------------------------------------------------------------
# 9. Leakage-prefix detection works
# ---------------------------------------------------------------------------


class TestDetectLeakage:
    def test_bracket_prefix_detected(self) -> None:
        issues = [_make_issue(1, title="[BUG] Something is broken")]
        result = detect_leakage(issues)
        assert 1 in result["prefix_matches"]

    def test_colon_prefix_detected(self) -> None:
        issues = [_make_issue(1, title="ENH: add new feature")]
        result = detect_leakage(issues)
        assert 1 in result["prefix_matches"]

    def test_clean_title_not_flagged(self) -> None:
        issues = [_make_issue(1, title="Memory leak in fit() method")]
        result = detect_leakage(issues)
        assert 1 not in result["prefix_matches"]
        assert 1 not in result["label_in_title"]

    def test_label_name_in_title_detected(self) -> None:
        issues = [
            _make_issue(
                1,
                title="This is a bug in the documentation",
                labels=["Bug"],
            )
        ]
        result = detect_leakage(issues, type_labels=PROVISIONAL_TYPE_LABELS)
        assert 1 in result["label_in_title"]

    def test_clean_title_with_bug_label_not_flagged_by_prefix(self) -> None:
        issues = [_make_issue(1, title="Unexpected NaN in output", labels=["Bug"])]
        result = detect_leakage(issues)
        assert 1 not in result["prefix_matches"]


# ---------------------------------------------------------------------------
# 10. Empty datasets produce a clear error
# ---------------------------------------------------------------------------


class TestLoadIssues:
    def test_empty_file_raises_value_error(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_issues(empty_file)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_issues(tmp_path / "does_not_exist.json")

    def test_non_array_json_raises_value_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load_issues(bad_file)

    def test_valid_file_loads_correctly(self, tmp_path: Path) -> None:
        issues = [_make_issue(1), _make_issue(2)]
        good_file = tmp_path / "issues.json"
        good_file.write_text(json.dumps(issues), encoding="utf-8")
        loaded = load_issues(good_file)
        assert len(loaded) == 2
        assert loaded[0]["number"] == 1


# ---------------------------------------------------------------------------
# Stage 1C — new tests
# ---------------------------------------------------------------------------


class TestComputeYearlyDistribution:
    def test_counts_by_year(self) -> None:
        issues = [
            _make_issue(1, created_at="2018-03-15T10:00:00Z"),
            _make_issue(2, created_at="2018-11-01T00:00:00Z"),
            _make_issue(3, created_at="2020-06-30T00:00:00Z"),
        ]
        yearly = compute_yearly_distribution(issues)
        assert yearly["2018"] == 2
        assert yearly["2020"] == 1
        assert yearly.get("2019", 0) == 0

    def test_missing_created_at_is_skipped(self) -> None:
        issues = [
            _make_issue(1, created_at=None),  # type: ignore[arg-type]
            _make_issue(2, created_at="2021-01-01T00:00:00Z"),
        ]
        yearly = compute_yearly_distribution(issues)
        assert yearly.get("2021") == 1
        assert sum(yearly.values()) == 1  # the None issue is not counted

    def test_empty_list_returns_empty_counter(self) -> None:
        yearly = compute_yearly_distribution([])
        assert len(yearly) == 0

    def test_returns_correct_year_strings(self) -> None:
        issues = [_make_issue(1, created_at="2023-12-31T23:59:59Z")]
        yearly = compute_yearly_distribution(issues)
        assert "2023" in yearly
        # Year keys should be exactly 4 characters
        for key in yearly:
            assert len(key) == 4
            assert key.isdigit()


class TestComputeLabelDistributionByYear:
    def _issues_for_year_test(self) -> list[dict]:
        return [
            _make_issue(1, labels=["Bug"], created_at="2019-05-01T00:00:00Z"),
            _make_issue(2, labels=["Bug"], created_at="2020-03-01T00:00:00Z"),
            _make_issue(3, labels=["Documentation"], created_at="2020-07-01T00:00:00Z"),
            _make_issue(4, labels=["New Feature"], created_at="2021-01-01T00:00:00Z"),
            # Multi-type: excluded by single-type rule
            _make_issue(
                5,
                labels=["Bug", "Documentation"],
                created_at="2021-02-01T00:00:00Z",
            ),
            # No type label: excluded
            _make_issue(6, labels=["Needs Triage"], created_at="2021-03-01T00:00:00Z"),
        ]

    def test_correct_year_assignment(self) -> None:
        issues = self._issues_for_year_test()
        labels = ["Bug", "Documentation", "New Feature"]
        by_year = compute_label_distribution_by_year(issues, labels)
        assert by_year["2019"]["Bug"] == 1
        assert by_year["2020"]["Bug"] == 1
        assert by_year["2020"]["Documentation"] == 1
        assert by_year["2021"]["New Feature"] == 1

    def test_multi_type_issue_not_counted(self) -> None:
        issues = self._issues_for_year_test()
        labels = ["Bug", "Documentation", "New Feature"]
        by_year = compute_label_distribution_by_year(issues, labels)
        # Issue 5 (Bug + Documentation in 2021) should not be counted
        assert by_year.get("2021", {}).get("Bug", 0) == 0
        assert by_year.get("2021", {}).get("Documentation", 0) == 0

    def test_non_type_labels_excluded(self) -> None:
        issues = self._issues_for_year_test()
        labels = ["Bug", "Documentation", "New Feature"]
        by_year = compute_label_distribution_by_year(issues, labels)
        # Issue 6 (Needs Triage, 2021) should not add any count
        assert sum(
            by_year.get("2021", {}).values()
        ) == 1  # only New Feature from issue 4


class TestBuildLabelSchemeStats:
    def _make_issues_for_scheme(self) -> list[dict]:
        return [
            _make_issue(1, labels=["Bug"]),
            _make_issue(2, labels=["Bug"]),
            _make_issue(3, labels=["Documentation"]),
            _make_issue(4, labels=["New Feature"]),
            _make_issue(5, labels=["RFC"]),
            _make_issue(6, labels=["Build / CI"]),
            _make_issue(7, labels=["Needs Triage"]),  # no scheme class
            # Multi-class: Bug maps to BugClass, RFC maps to Enhancement
            _make_issue(8, labels=["Bug", "RFC"]),
        ]

    def test_five_class_scheme_counts_correctly(self) -> None:
        issues = self._make_issues_for_scheme()
        scheme_a = {
            "Bug": ["Bug"],
            "Documentation": ["Documentation"],
            "New Feature": ["New Feature"],
            "RFC": ["RFC"],
            "Build / CI": ["Build / CI"],
        }
        stats = build_label_scheme_stats(issues, scheme_a)
        assert stats["class_counts"]["Bug"] == 2
        assert stats["class_counts"]["Documentation"] == 1
        assert stats["total_usable"] == 6  # issues 1-6 each map to one class
        assert stats["excluded_unlabelled"] == 1  # issue 7 (Needs Triage)
        assert stats["excluded_multi_class"] == 1  # issue 8 (Bug + RFC)

    def test_merged_scheme_increases_class_count(self) -> None:
        issues = self._make_issues_for_scheme()
        # Merge RFC into New Feature
        scheme_b = {
            "Bug": ["Bug"],
            "Documentation": ["Documentation"],
            "Enhancement": ["New Feature", "RFC"],
            "Build / CI": ["Build / CI"],
        }
        stats = build_label_scheme_stats(issues, scheme_b)
        assert stats["class_counts"]["Enhancement"] == 2  # New Feature + RFC
        assert stats["total_usable"] == 6  # issues 1-6 each map to one class
        # Issue 8 (Bug + RFC) maps to Bug + Enhancement → multi-class, excluded
        assert stats["excluded_multi_class"] == 1

    def test_imbalance_ratio_computed(self) -> None:
        issues = self._make_issues_for_scheme()
        scheme_a = {
            "Bug": ["Bug"],
            "Documentation": ["Documentation"],
            "New Feature": ["New Feature"],
            "RFC": ["RFC"],
            "Build / CI": ["Build / CI"],
        }
        stats = build_label_scheme_stats(issues, scheme_a)
        # max = 2 (Bug), min = 1 (others) → ratio = 2.0
        assert stats["imbalance_ratio"] == pytest.approx(2.0)

    def test_min_class_count_is_correct(self) -> None:
        issues = self._make_issues_for_scheme()
        scheme_b = {
            "Bug": ["Bug"],
            "Documentation": ["Documentation"],
            "Enhancement": ["New Feature", "RFC"],
            "Build / CI": ["Build / CI"],
        }
        stats = build_label_scheme_stats(issues, scheme_b)
        assert stats["min_class_count"] == 1

    def test_empty_scheme_excludes_all(self) -> None:
        issues = self._make_issues_for_scheme()
        stats = build_label_scheme_stats(issues, {})
        assert stats["total_usable"] == 0
        assert stats["excluded_unlabelled"] == len(issues)
