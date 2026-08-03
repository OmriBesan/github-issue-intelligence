"""
tests/test_combine.py
======================
Unit tests for the issue_intelligence.data.combine module.

Covers:
- Combining multiple raw files (load_and_combine)
- Deduplication across files
- Continuous yearly coverage detection (find_yearly_gaps)
- Label-review sampling (sample_issues_by_label)
- Deterministic sampling with a fixed random seed
- Candidate 3-class scheme (uses build_label_scheme_stats)
- Exclusion of Build / CI from the type target
- Detection of ambiguous multi-type issues
- build_cooccurrence_stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Make the src package importable without installation
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.audit import build_label_scheme_stats  # noqa: E402
from issue_intelligence.data.combine import (  # noqa: E402
    build_cooccurrence_stats,
    find_yearly_gaps,
    load_and_combine,
    sample_issues_by_label,
    save_combined,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    number: int,
    labels: list[str] | None = None,
    created_at: str = "2022-06-15T10:00:00Z",
    body: str = "Body text.",
) -> dict[str, Any]:
    return {
        "id": 90000 + number,
        "number": number,
        "title": f"Issue {number}",
        "body": body,
        "state": "open",
        "labels": labels or [],
        "created_at": created_at,
        "updated_at": created_at,
        "closed_at": None,
        "comments": 0,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "user_type": "User",
    }


def _write_issues(issues: list[dict], path: Path) -> None:
    path.write_text(json.dumps(issues), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. load_and_combine — basic combination
# ---------------------------------------------------------------------------


class TestLoadAndCombine:
    def test_combine_single_file(self, tmp_path: Path) -> None:
        issues = [_make_issue(1), _make_issue(2)]
        p = tmp_path / "a.json"
        _write_issues(issues, p)
        combined, stats = load_and_combine([p])
        assert len(combined) == 2
        assert stats["total_issues"] == 2
        assert stats["duplicates_found"] == 0

    def test_combine_two_non_overlapping_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_issues([_make_issue(1), _make_issue(2)], a)
        _write_issues([_make_issue(3), _make_issue(4)], b)
        combined, stats = load_and_combine([a, b])
        assert len(combined) == 4
        assert stats["duplicates_found"] == 0
        assert stats["files_loaded"] == 2

    def test_combine_maintains_all_unique_issues(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        issues_a = [_make_issue(i) for i in range(1, 6)]
        issues_b = [_make_issue(i) for i in range(6, 11)]
        _write_issues(issues_a, a)
        _write_issues(issues_b, b)
        combined, stats = load_and_combine([a, b])
        numbers = {i["number"] for i in combined}
        assert numbers == set(range(1, 11))

    def test_deduplication_across_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        # Issue 2 appears in both files
        _write_issues([_make_issue(1), _make_issue(2)], a)
        _write_issues([_make_issue(2), _make_issue(3)], b)
        combined, stats = load_and_combine([a, b])
        assert len(combined) == 3  # 1, 2, 3 — not 4
        assert stats["duplicates_found"] == 1

    def test_first_occurrence_wins_on_duplicate(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        issue_a = _make_issue(1, labels=["Bug"])
        issue_b = _make_issue(1, labels=["Documentation"])  # same ID, different labels
        _write_issues([issue_a], a)
        _write_issues([issue_b], b)
        combined, _ = load_and_combine([a, b])
        assert len(combined) == 1
        assert combined[0]["labels"] == ["Bug"]  # first file wins

    def test_sorted_by_date_after_combine(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_issues([_make_issue(1, created_at="2020-01-01T00:00:00Z")], a)
        _write_issues([_make_issue(2, created_at="2019-01-01T00:00:00Z")], b)
        combined, _ = load_and_combine([a, b], sort_by_date=True)
        assert combined[0]["number"] == 2  # 2019 before 2020
        assert combined[1]["number"] == 1

    def test_stats_per_file_counts(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_issues([_make_issue(1)], a)
        _write_issues([_make_issue(2), _make_issue(3)], b)
        _, stats = load_and_combine([a, b])
        assert stats["issues_per_file"][str(a)] == 1
        assert stats["issues_per_file"][str(b)] == 2


# ---------------------------------------------------------------------------
# 2. find_yearly_gaps
# ---------------------------------------------------------------------------


class TestFindYearlyGaps:
    def test_no_gaps_in_consecutive_years(self) -> None:
        issues = [
            _make_issue(i, created_at=f"{2020 + i}-01-01T00:00:00Z")
            for i in range(3)  # 2020, 2021, 2022
        ]
        info = find_yearly_gaps(issues, min_year=2020)
        assert info["gaps"] == []

    def test_detects_missing_year(self) -> None:
        issues = [
            _make_issue(1, created_at="2018-01-01T00:00:00Z"),
            _make_issue(2, created_at="2020-01-01T00:00:00Z"),  # 2019 missing
        ]
        info = find_yearly_gaps(issues, min_year=2018)
        assert "2019" in info["gaps"]

    def test_detects_multiple_gaps(self) -> None:
        issues = [
            _make_issue(1, created_at="2015-01-01T00:00:00Z"),
            _make_issue(2, created_at="2018-01-01T00:00:00Z"),
            # 2016, 2017 missing
        ]
        info = find_yearly_gaps(issues, min_year=2015)
        assert "2016" in info["gaps"]
        assert "2017" in info["gaps"]
        assert "2015" not in info["gaps"]
        assert "2018" not in info["gaps"]

    def test_empty_list_returns_no_gaps(self) -> None:
        info = find_yearly_gaps([])
        assert info["years_present"] == []
        assert info["gaps"] == []

    def test_explicit_min_year_extends_check(self) -> None:
        issues = [_make_issue(1, created_at="2018-01-01T00:00:00Z")]
        info = find_yearly_gaps(issues, min_year=2015)
        # 2015, 2016, 2017 should be gaps
        assert "2015" in info["gaps"]
        assert "2016" in info["gaps"]
        assert "2017" in info["gaps"]
        assert "2018" not in info["gaps"]


# ---------------------------------------------------------------------------
# 3. sample_issues_by_label
# ---------------------------------------------------------------------------


class TestSampleIssuesByLabel:
    def _make_many(self, label: str, n: int) -> list[dict]:
        return [_make_issue(i, labels=[label]) for i in range(n)]

    def test_sample_returns_at_most_n(self) -> None:
        issues = self._make_many("Bug", 100)
        sample = sample_issues_by_label(issues, "Bug", n=40)
        assert len(sample) <= 40

    def test_sample_returns_all_when_fewer_than_n(self) -> None:
        issues = self._make_many("RFC", 5)
        sample = sample_issues_by_label(issues, "RFC", n=40)
        assert len(sample) == 5

    def test_sample_is_deterministic_with_seed(self) -> None:
        issues = self._make_many("Bug", 100)
        sample_a = sample_issues_by_label(issues, "Bug", n=40, seed=42)
        sample_b = sample_issues_by_label(issues, "Bug", n=40, seed=42)
        assert [i["number"] for i in sample_a] == [i["number"] for i in sample_b]

    def test_different_seeds_give_different_samples(self) -> None:
        issues = self._make_many("Bug", 100)
        sample_a = sample_issues_by_label(issues, "Bug", n=40, seed=42)
        sample_b = sample_issues_by_label(issues, "Bug", n=40, seed=99)
        # Different seeds should (with very high probability) give different samples
        assert [i["number"] for i in sample_a] != [i["number"] for i in sample_b]

    def test_sample_returns_empty_for_absent_label(self) -> None:
        issues = self._make_many("Bug", 10)
        sample = sample_issues_by_label(issues, "RFC", n=40)
        assert sample == []

    def test_sample_excludes_issues_without_label(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug"]),
            _make_issue(2, labels=["Documentation"]),
            _make_issue(3, labels=["Bug"]),
        ]
        sample = sample_issues_by_label(issues, "Bug", n=40)
        assert all("Bug" in i["labels"] for i in sample)
        assert len(sample) == 2


# ---------------------------------------------------------------------------
# 4. Candidate 3-class scheme (using build_label_scheme_stats)
# ---------------------------------------------------------------------------

# The candidate scheme excludes Build / CI from the type target.
CANDIDATE_SCHEME = {
    "Bug": ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement": ["Enhancement", "New Feature", "RFC"],
}


class TestCandidateThreeClassScheme:
    def _make_dataset(self) -> list[dict]:
        return [
            _make_issue(1, labels=["Bug"]),
            _make_issue(2, labels=["Bug"]),
            _make_issue(3, labels=["Documentation"]),
            _make_issue(4, labels=["Enhancement"]),
            _make_issue(5, labels=["New Feature"]),
            _make_issue(6, labels=["RFC"]),
            _make_issue(7, labels=["Build / CI"]),  # excluded (not in scheme)
            _make_issue(8, labels=["help wanted"]),  # excluded (not in scheme)
            _make_issue(9, labels=["Bug", "Enhancement"]),  # multi-class
        ]

    def test_build_ci_excluded_from_three_class_scheme(self) -> None:
        issues = self._make_dataset()
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        # Build / CI (issue 7) must NOT appear in any class count
        for cls, cnt in stats["class_counts"].items():
            assert cls in CANDIDATE_SCHEME, f"Unexpected class: {cls}"
        # Issue 7 is excluded (no type target)
        assert stats["excluded_unlabelled"] >= 1

    def test_enhancement_merges_three_raw_labels(self) -> None:
        issues = self._make_dataset()
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        # Enhancement (4) + New Feature (5) + RFC (6) = 3
        assert stats["class_counts"]["Enhancement"] == 3

    def test_multi_type_issue_excluded(self) -> None:
        issues = self._make_dataset()
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        # Issue 9 (Bug + Enhancement) maps to Bug + Enhancement → excluded
        assert stats["excluded_multi_class"] >= 1

    def test_imbalance_ratio_below_threshold(self) -> None:
        # With larger balanced dataset the ratio should improve
        issues = (
            [_make_issue(i, labels=["Bug"]) for i in range(1, 11)]
            + [_make_issue(i, labels=["Documentation"]) for i in range(11, 21)]
            + [_make_issue(i, labels=["Enhancement"]) for i in range(21, 31)]
        )
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        assert stats["imbalance_ratio"] == pytest.approx(1.0)

    def test_total_usable_correct(self) -> None:
        issues = self._make_dataset()
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        # Usable: Bug(1) Bug(2) Doc(3) Enh(4) NF(5) RFC(6) = 6
        assert stats["total_usable"] == 6


# ---------------------------------------------------------------------------
# 5. Detection of ambiguous multi-type issues
# ---------------------------------------------------------------------------


class TestAmbiguousMultiTypeDetection:
    def test_enhancement_bug_issue_is_excluded(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "Enhancement"]),
        ]
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        assert stats["excluded_multi_class"] == 1
        assert stats["total_usable"] == 0

    def test_single_type_issue_is_not_ambiguous(self) -> None:
        issues = [
            _make_issue(1, labels=["Bug", "Easy"]),  # Easy is not a type label
        ]
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        assert stats["excluded_multi_class"] == 0
        assert stats["class_counts"]["Bug"] == 1

    def test_enhancement_new_feature_is_excluded(self) -> None:
        issues = [
            _make_issue(1, labels=["Enhancement", "New Feature"]),
        ]
        stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
        # Both "Enhancement" and "New Feature" map to the same class → OK
        # The issue has one effective class: Enhancement → NOT excluded
        assert stats["excluded_multi_class"] == 0
        assert stats["class_counts"]["Enhancement"] == 1


# ---------------------------------------------------------------------------
# 6. build_cooccurrence_stats
# ---------------------------------------------------------------------------


class TestBuildCooccurrenceStats:
    def test_cooccurrence_counts_correctly(self) -> None:
        issues = [
            _make_issue(1, labels=["Enhancement", "Bug"]),
            _make_issue(2, labels=["Enhancement"]),
            _make_issue(3, labels=["New Feature", "Documentation"]),
        ]
        stats = build_cooccurrence_stats(
            issues,
            target_labels=["Enhancement", "New Feature"],
            reference_labels=["Bug", "Documentation"],
        )
        assert stats["cooccurrence"]["Enhancement"]["Bug"] == 1
        assert stats["cooccurrence"]["Enhancement"].get("Documentation", 0) == 0
        assert stats["cooccurrence"]["New Feature"]["Documentation"] == 1
        assert stats["issues_with_target"]["Enhancement"] == 2
        assert stats["issues_with_target"]["New Feature"] == 1

    def test_no_cooccurrence_returns_zeros(self) -> None:
        issues = [_make_issue(1, labels=["Enhancement"])]
        stats = build_cooccurrence_stats(
            issues,
            target_labels=["Enhancement"],
            reference_labels=["Bug"],
        )
        assert stats["cooccurrence"]["Enhancement"].get("Bug", 0) == 0

    def test_total_issues_checked_is_full_dataset(self) -> None:
        issues = [_make_issue(i) for i in range(10)]
        stats = build_cooccurrence_stats(
            issues,
            target_labels=["Enhancement"],
            reference_labels=["Bug"],
        )
        assert stats["total_issues_checked"] == 10


# ---------------------------------------------------------------------------
# 7. save_combined (file I/O smoke test)
# ---------------------------------------------------------------------------


class TestSaveCombined:
    def test_save_creates_issues_file(self, tmp_path: Path) -> None:
        issues = [_make_issue(1), _make_issue(2)]
        output = tmp_path / "combined.json"
        save_combined(issues, {"key": "value"}, output)
        assert output.exists()

    def test_save_creates_metadata_file(self, tmp_path: Path) -> None:
        issues = [_make_issue(1)]
        output = tmp_path / "combined.json"
        save_combined(issues, {"key": "value"}, output)
        meta = tmp_path / "combined_metadata.json"
        assert meta.exists()

    def test_save_content_is_valid_json(self, tmp_path: Path) -> None:
        import json as json_lib

        issues = [_make_issue(1), _make_issue(2)]
        output = tmp_path / "combined.json"
        save_combined(issues, {"count": 2}, output)
        loaded = json_lib.loads(output.read_text(encoding="utf-8"))
        assert len(loaded) == 2
        assert loaded[0]["number"] == 1
