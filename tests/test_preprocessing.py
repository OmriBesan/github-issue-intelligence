"""
tests/test_preprocessing.py
============================
Unit tests for issue_intelligence.data.preprocessing.

All tests use synthetic issue records — no external files are read.

Coverage
--------
 1. Bug label mapping
 2. Documentation label mapping
 3. Enhancement mapping from raw label "Enhancement"
 4. Enhancement mapping from raw label "New Feature"
 5. Enhancement mapping from raw label "RFC"
 6. Build / CI does not create a target
 7. No-target issues are excluded
 8. Multi-target issues are excluded
 9. Duplicate IDs are removed deterministically (first wins)
10. Missing body handling (None → "")
11. Empty combined text is excluded
12. Prefix removal — colon forms (BUG:, ENH:, RFC:, DOC:, DOCS:, FEATURE:)
13. Prefix removal — bracket forms ([BUG], [ENH], [RFC], [DOCS])
14. Prefix removal — case-insensitive
15. Prefix word in middle of title is NOT removed
16. Raw title and body remain unchanged
17. Combined text is built correctly
18. CSV and JSONL records are equivalent
19. Metadata counts are correct
20. Output ordering is deterministic (created_at, then issue_id)
21. Empty input returns empty records and zero stats
22. NEW FEATURE: prefix is removed (multi-word form)
23. Enhancement + New Feature maps to one class (not excluded)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.preprocessing import (  # noqa: E402
    OUTPUT_COLUMNS,
    build_combined_text,
    build_dataset,
    build_metadata,
    clean_body,
    clean_title,
    map_labels_to_class,
    save_csv,
    save_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ID_COUNTER = 500_000  # Use high IDs to avoid collisions with real data


def _make_issue(
    labels: list[str],
    title: str = "A sample issue title",
    body: str = "Body text.",
    issue_id: int | None = None,
    number: int = 1,
    created_at: str = "2022-06-15T10:00:00Z",
) -> dict[str, Any]:
    global _ID_COUNTER
    _ID_COUNTER += 1
    return {
        "id": issue_id if issue_id is not None else _ID_COUNTER,
        "number": number,
        "title": title,
        "body": body,
        "state": "open",
        "labels": labels,
        "comments": 0,
        "created_at": created_at,
        "updated_at": created_at,
        "closed_at": None,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "user_type": "User",
    }


# ---------------------------------------------------------------------------
# 1-6: Label mapping
# ---------------------------------------------------------------------------


class TestLabelMapping:
    def test_bug_mapping(self) -> None:
        assert map_labels_to_class(["Bug"]) == "Bug"

    def test_documentation_mapping(self) -> None:
        assert map_labels_to_class(["Documentation"]) == "Documentation"

    def test_enhancement_from_enhancement(self) -> None:
        assert map_labels_to_class(["Enhancement"]) == "Enhancement"

    def test_enhancement_from_new_feature(self) -> None:
        assert map_labels_to_class(["New Feature"]) == "Enhancement"

    def test_enhancement_from_rfc(self) -> None:
        assert map_labels_to_class(["RFC"]) == "Enhancement"

    def test_build_ci_returns_none(self) -> None:
        """Build / CI must never produce a target class."""
        assert map_labels_to_class(["Build / CI"]) is None

    def test_empty_labels_returns_none(self) -> None:
        assert map_labels_to_class([]) is None

    def test_only_non_type_labels_returns_none(self) -> None:
        assert map_labels_to_class(["help wanted", "good first issue"]) is None

    def test_multi_target_returns_none(self) -> None:
        """Issues mapping to Bug AND Documentation are excluded."""
        assert map_labels_to_class(["Bug", "Documentation"]) is None

    def test_same_class_multiple_raw_labels_is_not_multi_target(self) -> None:
        """Enhancement + New Feature both map to 'Enhancement' → still one class."""
        result = map_labels_to_class(["Enhancement", "New Feature"])
        assert result == "Enhancement"

    def test_build_ci_with_type_label_excluded(self) -> None:
        """Bug + Build/CI still maps to exactly Bug (Build/CI ignored as non-type)."""
        result = map_labels_to_class(["Bug", "Build / CI"])
        assert result == "Bug"


# ---------------------------------------------------------------------------
# 7-11: Exclusion logic in build_dataset
# ---------------------------------------------------------------------------


class TestExclusionLogic:
    def test_no_target_issues_excluded(self) -> None:
        issues = [_make_issue(labels=[])]
        records, stats = build_dataset(issues)
        assert len(records) == 0
        assert stats["excluded_no_target"] == 1

    def test_build_ci_issues_excluded(self) -> None:
        issues = [_make_issue(labels=["Build / CI"])]
        records, stats = build_dataset(issues)
        assert len(records) == 0
        assert stats["excluded_no_target"] == 1

    def test_multi_target_issues_excluded(self) -> None:
        issues = [_make_issue(labels=["Bug", "Documentation"])]
        records, stats = build_dataset(issues)
        assert len(records) == 0
        assert stats["excluded_multi_target"] == 1

    def test_duplicate_ids_removed_first_wins(self) -> None:
        """Only the first occurrence of a duplicate issue ID is kept."""
        issues = [
            _make_issue(labels=["Bug"], title="First", issue_id=99001),
            _make_issue(labels=["Bug"], title="Second", issue_id=99001),
        ]
        records, stats = build_dataset(issues)
        assert len(records) == 1
        assert records[0]["raw_title"] == "First"
        assert stats["excluded_duplicate"] == 1

    def test_empty_combined_text_excluded(self) -> None:
        """Issue with empty title AND empty body must be excluded."""
        issues = [_make_issue(labels=["Bug"], title="", body="")]
        records, stats = build_dataset(issues)
        assert len(records) == 0
        assert stats["excluded_empty_text"] == 1

    def test_whitespace_only_combined_text_excluded(self) -> None:
        """Title and body containing only whitespace → excluded."""
        issues = [_make_issue(labels=["Bug"], title="   ", body="   ")]
        records, stats = build_dataset(issues)
        assert len(records) == 0
        assert stats["excluded_empty_text"] == 1

    def test_missing_body_does_not_exclude(self) -> None:
        """None body becomes "" but title alone keeps the issue usable."""
        issues = [
            _make_issue(labels=["Documentation"], title="Update readme", body=None)
        ]
        records, stats = build_dataset(issues)
        assert len(records) == 1
        assert records[0]["raw_body"] == ""
        assert records[0]["clean_body"] == ""


# ---------------------------------------------------------------------------
# 12-15: Text cleaning — prefix removal
# ---------------------------------------------------------------------------


class TestPrefixRemoval:
    def _clean(self, title: str) -> tuple[str, bool]:
        return clean_title(title)

    # --- colon forms ---
    def test_bug_colon_removed(self) -> None:
        clean, removed = self._clean("BUG: Something is wrong")
        assert clean == "Something is wrong"
        assert removed is True

    def test_enhancements_colon_removed(self) -> None:
        clean, removed = self._clean("ENH: Add new widget")
        assert clean == "Add new widget"
        assert removed is True

    def test_enhancement_full_colon_removed(self) -> None:
        clean, removed = self._clean("ENHANCEMENT: Better docs")
        assert clean == "Better docs"
        assert removed is True

    def test_rfc_colon_removed(self) -> None:
        clean, removed = self._clean("RFC: Redesign the API")
        assert clean == "Redesign the API"
        assert removed is True

    def test_doc_colon_removed(self) -> None:
        clean, removed = self._clean("DOC: Fix typo")
        assert clean == "Fix typo"
        assert removed is True

    def test_docs_colon_removed(self) -> None:
        clean, removed = self._clean("DOCS: Update tutorial")
        assert clean == "Update tutorial"
        assert removed is True

    def test_feature_colon_removed(self) -> None:
        clean, removed = self._clean("FEATURE: Support new format")
        assert clean == "Support new format"
        assert removed is True

    def test_new_feature_colon_removed(self) -> None:
        clean, removed = self._clean("NEW FEATURE: Export to PDF")
        assert clean == "Export to PDF"
        assert removed is True

    # --- bracket forms ---
    def test_bracket_bug_removed(self) -> None:
        clean, removed = self._clean("[BUG] KeyError when using sparse matrix")
        assert clean == "KeyError when using sparse matrix"
        assert removed is True

    def test_bracket_enh_removed(self) -> None:
        clean, removed = self._clean("[ENH] Add progress bar")
        assert clean == "Add progress bar"
        assert removed is True

    def test_bracket_rfc_removed(self) -> None:
        clean, removed = self._clean("[RFC] New output_format argument")
        assert clean == "New output_format argument"
        assert removed is True

    def test_bracket_docs_removed(self) -> None:
        clean, removed = self._clean("[DOCS] Clarify fit() signature")
        assert clean == "Clarify fit() signature"
        assert removed is True

    # --- case insensitivity ---
    def test_lowercase_prefix_removed(self) -> None:
        clean, removed = self._clean("bug: unexpected behaviour in fit()")
        assert clean == "unexpected behaviour in fit()"
        assert removed is True

    def test_mixed_case_prefix_removed(self) -> None:
        clean, removed = self._clean("Bug: unexpected behaviour in fit()")
        assert clean == "unexpected behaviour in fit()"
        assert removed is True

    def test_bracket_lowercase_removed(self) -> None:
        clean, removed = self._clean("[bug] regression in v1.2")
        assert clean == "regression in v1.2"
        assert removed is True

    # --- prefix in middle NOT removed ---
    def test_prefix_in_middle_not_removed(self) -> None:
        title = "Improve performance of BUG: detection"
        clean, removed = self._clean(title)
        assert removed is False
        assert "BUG" in clean

    def test_bug_word_in_sentence_not_removed(self) -> None:
        title = "Fix the bug in LinearSVC"
        clean, removed = self._clean(title)
        assert removed is False
        assert clean == title

    def test_no_prefix_no_change(self) -> None:
        title = "Something completely normal"
        clean, removed = self._clean(title)
        assert clean == title
        assert removed is False


# ---------------------------------------------------------------------------
# 16-17: Raw fields preserved, combined text correct
# ---------------------------------------------------------------------------


class TestRawPreservationAndCombined:
    def test_raw_title_unchanged(self) -> None:
        """The raw_title field in the record must match the original exactly."""
        original = "BUG: Something is wrong"
        issues = [_make_issue(labels=["Bug"], title=original)]
        records, _ = build_dataset(issues)
        assert records[0]["raw_title"] == original

    def test_raw_body_unchanged(self) -> None:
        """The raw_body field must match the original body exactly."""
        original_body = "  Some body with   spaces  "
        issues = [_make_issue(labels=["Bug"], body=original_body)]
        records, _ = build_dataset(issues)
        assert records[0]["raw_body"] == original_body

    def test_clean_title_differs_from_raw_after_prefix_removal(self) -> None:
        issues = [_make_issue(labels=["Bug"], title="BUG: Real title")]
        records, _ = build_dataset(issues)
        assert records[0]["raw_title"] == "BUG: Real title"
        assert records[0]["clean_title"] == "Real title"

    def test_combined_text_uses_double_newline_separator(self) -> None:
        result = build_combined_text("Title", "Body")
        assert result == "Title\n\nBody"

    def test_combined_text_only_title_when_body_empty(self) -> None:
        result = build_combined_text("Title", "")
        assert result == "Title"

    def test_combined_text_only_body_when_title_empty(self) -> None:
        result = build_combined_text("", "Body")
        assert result == "Body"

    def test_combined_text_in_record(self) -> None:
        issues = [_make_issue(labels=["Bug"], title="My title", body="My body")]
        records, _ = build_dataset(issues)
        assert records[0]["combined_text"] == "My title\n\nMy body"

    def test_combined_text_after_prefix_removal(self) -> None:
        issues = [_make_issue(labels=["Bug"], title="BUG: My title", body="My body")]
        records, _ = build_dataset(issues)
        assert records[0]["combined_text"] == "My title\n\nMy body"

    def test_length_fields_match_cleaned_strings(self) -> None:
        issues = [_make_issue(labels=["Bug"], title="Hello world", body="Test body")]
        records, _ = build_dataset(issues)
        r = records[0]
        assert r["title_length"] == len(r["clean_title"])
        assert r["body_length"] == len(r["clean_body"])
        assert r["combined_length"] == len(r["combined_text"])


# ---------------------------------------------------------------------------
# 18: CSV and JSONL equivalence
# ---------------------------------------------------------------------------


class TestIOEquivalence:
    def _make_records(self) -> list[dict]:
        issues = [
            _make_issue(labels=["Bug"], title="Bug title", body="Bug body"),
            _make_issue(labels=["Documentation"], title="Doc title", body="Doc body"),
            _make_issue(labels=["Enhancement"], title="Enh title", body="Enh body"),
        ]
        records, _ = build_dataset(issues)
        return records

    def test_csv_row_count_matches_records(self, tmp_path: Path) -> None:
        records = self._make_records()
        csv_path = tmp_path / "out.csv"
        save_csv(records, csv_path)
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(records)

    def test_jsonl_line_count_matches_records(self, tmp_path: Path) -> None:
        records = self._make_records()
        jsonl_path = tmp_path / "out.jsonl"
        save_jsonl(records, jsonl_path)
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(records)

    def test_csv_and_jsonl_same_targets(self, tmp_path: Path) -> None:
        records = self._make_records()
        csv_path = tmp_path / "out.csv"
        jsonl_path = tmp_path / "out.jsonl"
        save_csv(records, csv_path)
        save_jsonl(records, jsonl_path)

        with csv_path.open(encoding="utf-8") as fh:
            csv_targets = [row["target"] for row in csv.DictReader(fh)]
        jsonl_targets = [
            json.loads(line)["target"]
            for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert csv_targets == jsonl_targets

    def test_csv_and_jsonl_same_issue_ids(self, tmp_path: Path) -> None:
        records = self._make_records()
        csv_path = tmp_path / "out.csv"
        jsonl_path = tmp_path / "out.jsonl"
        save_csv(records, csv_path)
        save_jsonl(records, jsonl_path)

        with csv_path.open(encoding="utf-8") as fh:
            csv_ids = [row["issue_id"] for row in csv.DictReader(fh)]
        jsonl_ids = [
            str(json.loads(line)["issue_id"])
            for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert csv_ids == jsonl_ids

    def test_all_output_columns_present_in_csv(self, tmp_path: Path) -> None:
        records = self._make_records()
        csv_path = tmp_path / "out.csv"
        save_csv(records, csv_path)
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert set(OUTPUT_COLUMNS) == set(reader.fieldnames or [])

    def test_all_output_columns_present_in_jsonl(self, tmp_path: Path) -> None:
        records = self._make_records()
        jsonl_path = tmp_path / "out.jsonl"
        save_jsonl(records, jsonl_path)
        first_line = jsonl_path.read_text(encoding="utf-8").splitlines()[0]
        row = json.loads(first_line)
        assert set(OUTPUT_COLUMNS) == set(row.keys())

    def test_internal_fields_not_in_csv(self, tmp_path: Path) -> None:
        records = self._make_records()
        csv_path = tmp_path / "out.csv"
        save_csv(records, csv_path)
        with csv_path.open(encoding="utf-8") as fh:
            fieldnames = csv.DictReader(fh).fieldnames or []
        assert "_prefix_removed" not in fieldnames

    def test_internal_fields_not_in_jsonl(self, tmp_path: Path) -> None:
        records = self._make_records()
        jsonl_path = tmp_path / "out.jsonl"
        save_jsonl(records, jsonl_path)
        first = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
        assert "_prefix_removed" not in first


# ---------------------------------------------------------------------------
# 19: Metadata counts
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_class_counts_correct(self, tmp_path: Path) -> None:
        issues = [
            _make_issue(labels=["Bug"]),
            _make_issue(labels=["Bug"]),
            _make_issue(labels=["Documentation"]),
            _make_issue(labels=["Enhancement"]),
            _make_issue(labels=[]),  # excluded
        ]
        records, stats = build_dataset(issues)
        csv_p = tmp_path / "out.csv"
        jsonl_p = tmp_path / "out.jsonl"
        save_csv(records, csv_p)
        save_jsonl(records, jsonl_p)
        meta = build_metadata(stats, records, Path("dummy.json"), csv_p, jsonl_p)

        assert meta["class_counts"]["Bug"] == 2
        assert meta["class_counts"]["Documentation"] == 1
        assert meta["class_counts"]["Enhancement"] == 1
        assert meta["excluded_no_target"] == 1
        assert meta["usable_issues"] == 4

    def test_metadata_total_raw_correct(self, tmp_path: Path) -> None:
        issues = [_make_issue(labels=["Bug"]) for _ in range(5)]
        records, stats = build_dataset(issues)
        csv_p = tmp_path / "out.csv"
        jsonl_p = tmp_path / "out.jsonl"
        save_csv(records, csv_p)
        save_jsonl(records, jsonl_p)
        meta = build_metadata(stats, records, Path("dummy.json"), csv_p, jsonl_p)
        assert meta["total_raw_issues"] == 5

    def test_metadata_imbalance_ratio_symmetric(self, tmp_path: Path) -> None:
        """2 Bug vs 2 Documentation → ratio 1.0."""
        issues = [
            _make_issue(labels=["Bug"]),
            _make_issue(labels=["Bug"]),
            _make_issue(labels=["Documentation"]),
            _make_issue(labels=["Documentation"]),
            _make_issue(labels=["Enhancement"]),
            _make_issue(labels=["Enhancement"]),
        ]
        records, stats = build_dataset(issues)
        assert stats["imbalance_ratio"] == pytest.approx(1.0)

    def test_metadata_contains_label_scheme(self, tmp_path: Path) -> None:
        issues = [_make_issue(labels=["Bug"])]
        records, stats = build_dataset(issues)
        csv_p = tmp_path / "out.csv"
        jsonl_p = tmp_path / "out.jsonl"
        save_csv(records, csv_p)
        save_jsonl(records, jsonl_p)
        meta = build_metadata(stats, records, Path("dummy.json"), csv_p, jsonl_p)
        assert "label_scheme" in meta
        assert "Enhancement" in meta["label_scheme"]


# ---------------------------------------------------------------------------
# 20: Output ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_records_sorted_by_created_at_ascending(self) -> None:
        issues = [
            _make_issue(labels=["Bug"], created_at="2022-03-01T00:00:00Z"),
            _make_issue(labels=["Bug"], created_at="2020-01-01T00:00:00Z"),
            _make_issue(labels=["Bug"], created_at="2021-06-15T00:00:00Z"),
        ]
        records, _ = build_dataset(issues)
        dates = [r["created_at"] for r in records]
        assert dates == sorted(dates)

    def test_records_stable_within_same_date(self) -> None:
        """Issues with same date are ordered by issue_id."""
        ts = "2022-06-15T00:00:00Z"
        issues = [
            _make_issue(labels=["Bug"], created_at=ts, issue_id=200),
            _make_issue(labels=["Bug"], created_at=ts, issue_id=100),
        ]
        records, _ = build_dataset(issues)
        ids = [r["issue_id"] for r in records]
        assert ids == sorted(ids)

    def test_deterministic_across_runs(self) -> None:
        """Calling build_dataset twice on the same input returns same order."""
        issues = [_make_issue(labels=["Bug"]) for _ in range(10)]
        r1, _ = build_dataset(issues)
        r2, _ = build_dataset(issues)
        assert [r["issue_id"] for r in r1] == [r["issue_id"] for r in r2]


# ---------------------------------------------------------------------------
# 21: Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_input_returns_empty_records(self) -> None:
        records, stats = build_dataset([])
        assert records == []

    def test_empty_input_stats_all_zero(self) -> None:
        _, stats = build_dataset([])
        assert stats["total_raw"] == 0
        assert stats["usable"] == 0
        assert stats["excluded_no_target"] == 0
        assert stats["excluded_multi_target"] == 0
        assert stats["excluded_duplicate"] == 0
        assert stats["excluded_empty_text"] == 0


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_new_feature_prefix_removed(self) -> None:
        clean, removed = clean_title("NEW FEATURE: Better export")
        assert clean == "Better export"
        assert removed is True

    def test_enhancement_new_feature_same_class_not_excluded(self) -> None:
        """Both labels map to Enhancement → exactly one class → usable."""
        issues = [_make_issue(labels=["Enhancement", "New Feature"])]
        records, stats = build_dataset(issues)
        assert len(records) == 1
        assert records[0]["target"] == "Enhancement"
        assert stats["excluded_multi_target"] == 0

    def test_clean_body_preserves_newlines(self) -> None:
        body = "Line 1\nLine 2\n\nLine 3"
        result = clean_body(body)
        assert "\n" in result
        assert "Line 1" in result
        assert "Line 3" in result

    def test_clean_body_preserves_code_block(self) -> None:
        body = "See this:\n```python\nimport numpy\n```\nEnd."
        result = clean_body(body)
        assert "```python" in result
        assert "import numpy" in result

    def test_missing_title_becomes_empty_string(self) -> None:
        clean, _ = clean_title(None)
        assert clean == ""

    def test_prefix_removal_count_in_stats(self) -> None:
        issues = [
            _make_issue(labels=["Bug"], title="BUG: Something"),
            _make_issue(labels=["Documentation"], title="DOC: Fix typo"),
            _make_issue(labels=["Enhancement"], title="Normal title"),
        ]
        _, stats = build_dataset(issues)
        assert stats["prefix_removals"] == 2
