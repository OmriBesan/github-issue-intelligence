"""
issue_intelligence.data.preprocessing
=======================================
Preprocessing pipeline that converts raw GitHub issue JSON records into a
clean, analysis-ready classification dataset.

Label mapping (Stage 2A — approved 2026-07-28)
-----------------------------------------------
Bug           <- ["Bug"]
Documentation <- ["Documentation"]
Enhancement   <- ["Enhancement", "New Feature", "RFC"]

Build / CI is explicitly excluded from the type target.  A "Build / CI" issue
almost always co-occurs with Bug (19.7% co-occurrence in the combined dataset)
and functions as a component modifier rather than an issue type.

This mapping was determined by:
  1. Historical label analysis across 12,190 combined issues (Stages 1B-1D).
  2. Co-occurrence statistics (label_review_sample.py output).
  3. Human inspection of reports/label_mapping_review.csv (160 rows, 40/label).
  4. Recorded in docs/decisions.md as Decisions 020 and 021.

Design principles
-----------------
* Conservative preprocessing: no stemming, no stopword removal, no lowercasing,
  no punctuation stripping.  Only structural noise (category prefixes, excess
  whitespace) is removed.
* Raw title and body are always preserved unchanged in the output.
* Code snippets, stack traces, and Markdown in the body are kept intact.
* Deterministic: fixed sort order ensures reproducible output from identical input.

Public API
----------
clean_title(raw)                -> (clean_title_str, prefix_removed_bool)
clean_body(raw)                 -> clean_body_str
build_combined_text(title, body)-> combined_str
map_labels_to_class(labels)     -> class_name | None
process_record(issue)           -> record_dict | None
build_dataset(issues)           -> (records, stats)
save_csv(records, path)
save_jsonl(records, path)
build_metadata(stats, records, source_path, csv_path, jsonl_path) -> dict
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the src package importable when called directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Label mapping constants
# ---------------------------------------------------------------------------

#: Approved three-class target scheme (Stage 2A, Decision 021).
LABEL_SCHEME: dict[str, list[str]] = {
    "Bug": ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement": ["Enhancement", "New Feature", "RFC"],
}

#: Reverse lookup: raw label name -> target class name.
RAW_LABEL_TO_CLASS: dict[str, str] = {
    raw_label: cls_name
    for cls_name, raw_labels in LABEL_SCHEME.items()
    for raw_label in raw_labels
}

#: Ordered list of output columns written to CSV and JSONL.
OUTPUT_COLUMNS: list[str] = [
    "issue_id",
    "issue_number",
    "created_at",
    "state",
    "comments",
    "html_url",
    "original_labels",
    "target",
    "raw_title",
    "raw_body",
    "clean_title",
    "clean_body",
    "combined_text",
    "title_length",
    "body_length",
    "combined_length",
]

# ---------------------------------------------------------------------------
# Prefix removal
# ---------------------------------------------------------------------------

# Leading category prefixes to strip from issue titles.
# Must match at the very beginning of the string (case-insensitive).
#
# Supported forms:
#   Colon form:   BUG:    ENH:    RFC:    DOC:    DOCS:   FEATURE:
#                 ENHANCEMENT:    NEW FEATURE:    FIX:    MAINT:
#   Bracket form: [BUG]   [ENH]   [RFC]   [DOC]   [DOCS]  [FEATURE]
#                 [ENHANCEMENT]   [NEW FEATURE]   [FIX]   [MAINT]
#
# NOT removed:
#   - The same word appearing in the middle of a title.
#   - Words followed only by another word (e.g. "Bug report:" is NOT removed
#     because "Bug" is not directly followed by a colon or bracket).

_PREFIX_PATTERN = re.compile(
    r"^(?:"
    # --- bracket forms ---
    r"\[BUG\][:\s-]*|"
    r"\[DOCS?\][:\s-]*|"
    r"\[ENH(?:ANCEMENT)?\][:\s-]*|"
    r"\[FEATURE\][:\s-]*|"
    r"\[NEW\s+FEATURE\][:\s-]*|"
    r"\[RFC\][:\s-]*|"
    r"\[FIX\][:\s-]*|"
    r"\[MAINT(?:ENANCE)?\][:\s-]*|"
    r"\[TST\][:\s-]*|"
    # --- colon forms ---
    r"BUG\s*:\s*|"
    r"DOCS?\s*:\s*|"
    r"ENH(?:ANCEMENT)?\s*:\s*|"
    r"FEATURE\s*:\s*|"
    r"NEW\s+FEATURE\s*:\s*|"
    r"RFC\s*:\s*|"
    r"FIX\s*:\s*|"
    r"MAINT(?:ENANCE)?\s*:\s*|"
    r"TST\s*:\s*"
    r")\s*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Whitespace normalisation helpers (internal)
# ---------------------------------------------------------------------------

_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def _normalize_whitespace(text: str) -> str:
    """
    Normalize horizontal whitespace without collapsing newlines.

    - Collapses runs of spaces/tabs within each line to a single space.
    - Collapses runs of 3+ consecutive blank lines to 2 blank lines.
    - Strips leading and trailing whitespace from the whole string.

    Newlines are preserved so that code blocks and multi-paragraph bodies
    remain readable and structurally intact.
    """
    if not text:
        return text
    lines = [_HORIZONTAL_SPACE.sub(" ", line) for line in text.split("\n")]
    joined = "\n".join(lines)
    joined = _EXCESS_BLANK_LINES.sub("\n\n", joined)
    return joined.strip()


# ---------------------------------------------------------------------------
# Public text-cleaning functions
# ---------------------------------------------------------------------------


def clean_title(raw: str | None) -> tuple[str, bool]:
    """
    Produce a clean issue title.

    Steps:
    1. Coerce ``None`` to ``""``.
    2. Strip leading/trailing whitespace and normalize horizontal spaces.
    3. Remove a known leading category prefix if present.

    Parameters
    ----------
    raw : str or None
        The original issue title value.

    Returns
    -------
    clean : str
        The cleaned title (may be equal to the stripped raw if no prefix found).
    prefix_removed : bool
        ``True`` if a category prefix was stripped from the beginning.
    """
    s = str(raw).strip() if raw is not None else ""
    s = _normalize_whitespace(s)
    cleaned = _PREFIX_PATTERN.sub("", s).strip()
    prefix_removed = cleaned != s
    return cleaned, prefix_removed


def clean_body(raw: str | None) -> str:
    """
    Produce a clean issue body.

    Steps:
    1. Coerce ``None`` to ``""``.
    2. Strip leading/trailing whitespace.
    3. Normalize horizontal whitespace within each line.
    4. Collapse excessive blank lines.

    Code snippets, stack traces, and Markdown formatting are preserved.
    No stopwords, stemming, lemmatization, lowercasing, or punctuation
    removal is applied.

    Parameters
    ----------
    raw : str or None
        The original issue body value.

    Returns
    -------
    str
        The cleaned body text.
    """
    s = str(raw).strip() if raw is not None else ""
    return _normalize_whitespace(s)


def build_combined_text(title: str, body: str) -> str:
    """
    Build combined text from a clean title and clean body.

    combined_text = clean_title + "\\n\\n" + clean_body

    If either part is empty it is omitted and no separator is added.

    Parameters
    ----------
    title : str
        Clean title (may be empty).
    body : str
        Clean body (may be empty).

    Returns
    -------
    str
        Combined text, or ``""`` if both inputs are empty.
    """
    parts = [p for p in (title, body) if p]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def map_labels_to_class(labels: list[str]) -> str | None:
    """
    Map a list of raw GitHub label names to a single target class.

    Parameters
    ----------
    labels : list of str
        Raw label names from the issue record.

    Returns
    -------
    str
        Target class name if exactly one class matches.
    None
        If zero classes match (unlabelled / non-type label) or more than
        one class matches (ambiguous multi-type issue).
    """
    matched_classes = {
        RAW_LABEL_TO_CLASS[lb] for lb in labels if lb in RAW_LABEL_TO_CLASS
    }
    if len(matched_classes) == 1:
        return next(iter(matched_classes))
    return None


# ---------------------------------------------------------------------------
# Single-record processing
# ---------------------------------------------------------------------------


def process_record(issue: dict) -> dict | None:
    """
    Convert a single raw GitHub issue dict into a processed record.

    Applies label mapping, text cleaning, and combined-text construction.

    Returns
    -------
    dict
        Processed record containing all ``OUTPUT_COLUMNS`` fields plus the
        internal ``_prefix_removed`` flag (stripped before I/O).
    None
        If the issue should be excluded:
        - label maps to zero or multiple target classes, OR
        - combined text is empty after cleaning.
    """
    labels: list[str] = issue.get("labels") or []
    target = map_labels_to_class(labels)
    if target is None:
        return None

    raw_t: str = issue.get("title") or ""
    raw_b: str = issue.get("body") or ""

    clean_t, prefix_removed = clean_title(raw_t)
    clean_b = clean_body(raw_b)
    combined = build_combined_text(clean_t, clean_b)

    if not combined.strip():
        return None

    return {
        "issue_id": issue.get("id"),
        "issue_number": issue.get("number"),
        "created_at": issue.get("created_at") or "",
        "state": issue.get("state") or "",
        "comments": issue.get("comments", 0),
        "html_url": issue.get("html_url") or "",
        "original_labels": json.dumps(labels, ensure_ascii=False),
        "target": target,
        "raw_title": raw_t,
        "raw_body": raw_b,
        "clean_title": clean_t,
        "clean_body": clean_b,
        "combined_text": combined,
        "title_length": len(clean_t),
        "body_length": len(clean_b),
        "combined_length": len(combined),
        "_prefix_removed": prefix_removed,  # internal; excluded from I/O
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    issues: list[dict],
) -> tuple[list[dict], dict]:
    """
    Process a list of raw issue dicts into a clean modelling dataset.

    Exclusion rules applied in order:
    1. Duplicate issue IDs — first occurrence kept, subsequent ones excluded.
    2. No target class — issue labels do not map to any known class.
    3. Multiple target classes — issue maps to more than one class (ambiguous).
    4. Empty combined text — both title and body are empty after cleaning.

    Records are sorted deterministically by (created_at, issue_id) ascending.

    Parameters
    ----------
    issues : list of dict
        Raw GitHub issue records (e.g. loaded from the combined JSON file).

    Returns
    -------
    records : list of dict
        Processed records containing all OUTPUT_COLUMNS plus ``_prefix_removed``.
    stats : dict
        Exclusion and class statistics.
    """
    total_raw = len(issues)
    excluded_no_target = 0
    excluded_multi_target = 0
    excluded_duplicate = 0
    excluded_empty_text = 0
    prefix_removals = 0

    seen_ids: set = set()
    records: list[dict] = []

    for issue in issues:
        issue_id = issue.get("id")

        # 1. Deduplication
        if issue_id in seen_ids:
            excluded_duplicate += 1
            continue
        if issue_id is not None:
            seen_ids.add(issue_id)

        # 2+3. Label mapping check
        labels: list[str] = issue.get("labels") or []
        matched_classes = {
            RAW_LABEL_TO_CLASS[lb] for lb in labels if lb in RAW_LABEL_TO_CLASS
        }
        if len(matched_classes) == 0:
            excluded_no_target += 1
            continue
        if len(matched_classes) > 1:
            excluded_multi_target += 1
            continue

        # 4. Text processing + empty combined text check
        rec = process_record(issue)
        if rec is None:
            # process_record returns None only for empty combined text here
            # (label mapping already passed above)
            excluded_empty_text += 1
            continue

        if rec["_prefix_removed"]:
            prefix_removals += 1
        records.append(rec)

    # Deterministic sort: created_at ascending, then issue_id ascending
    records.sort(
        key=lambda r: (r.get("created_at") or "", r.get("issue_id") or 0)
    )

    # Class counts
    class_counts: dict[str, int] = {cls: 0 for cls in LABEL_SCHEME}
    for rec in records:
        class_counts[rec["target"]] += 1

    usable = len(records)
    counts = list(class_counts.values())
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = round(max_count / min_count, 3) if min_count > 0 else None

    stats = {
        "total_raw": total_raw,
        "excluded_no_target": excluded_no_target,
        "excluded_multi_target": excluded_multi_target,
        "excluded_duplicate": excluded_duplicate,
        "excluded_empty_text": excluded_empty_text,
        "prefix_removals": prefix_removals,
        "usable": usable,
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
    }

    return records, stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_csv(records: list[dict], output_path: Path | str) -> None:
    """
    Save records to a UTF-8 CSV file.

    Internal fields (names beginning with ``_``) are excluded from output.

    Parameters
    ----------
    records : list of dict
    output_path : Path or str
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_jsonl(records: list[dict], output_path: Path | str) -> None:
    """
    Save records to a UTF-8 JSONL file (one JSON object per line).

    Internal fields (names beginning with ``_``) are excluded from output.

    Parameters
    ----------
    records : list of dict
    output_path : Path or str
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            row = {k: rec[k] for k in OUTPUT_COLUMNS}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(
    stats: dict,
    records: list[dict],
    source_path: Path | str,
    output_csv: Path | str,
    output_jsonl: Path | str,
) -> dict:
    """
    Build the provenance and statistics metadata dict.

    Parameters
    ----------
    stats : dict
        Output from ``build_dataset()``.
    records : list of dict
        Processed records.
    source_path : Path or str
        Path to the raw combined JSON input file.
    output_csv : Path or str
        Path to the saved CSV output file.
    output_jsonl : Path or str
        Path to the saved JSONL output file.

    Returns
    -------
    dict
        Metadata suitable for JSON serialisation.
    """
    csv_p = Path(output_csv)
    jsonl_p = Path(output_jsonl)

    dates = [r["created_at"] for r in records if r.get("created_at")]
    usable = stats["usable"]
    class_counts = stats["class_counts"]

    class_pcts = {
        cls: round(100 * cnt / usable, 2) if usable else 0.0
        for cls, cnt in class_counts.items()
    }

    return {
        "source_file": str(source_path),
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "label_scheme": LABEL_SCHEME,
        "label_scheme_rationale": (
            "Enhancement / New Feature / RFC merged into one Enhancement class "
            "based on historical label evolution analysis (Stages 1B-1D). "
            "Build / CI excluded because it is a component modifier, not a type. "
            "See docs/decisions.md Decision 020-021 and "
            "reports/label_mapping_review.csv."
        ),
        "total_raw_issues": stats["total_raw"],
        "excluded_no_target": stats["excluded_no_target"],
        "excluded_multi_target": stats["excluded_multi_target"],
        "excluded_duplicate": stats["excluded_duplicate"],
        "excluded_empty_text": stats["excluded_empty_text"],
        "usable_issues": usable,
        "class_counts": class_counts,
        "class_percentages": class_pcts,
        "imbalance_ratio": stats["imbalance_ratio"],
        "prefix_removals": stats["prefix_removals"],
        "prefix_removal_pct": (
            round(100 * stats["prefix_removals"] / usable, 2) if usable else 0.0
        ),
        "earliest_created_at": min(dates) if dates else None,
        "latest_created_at": max(dates) if dates else None,
        "output_csv_sha256": _sha256(csv_p) if csv_p.exists() else None,
        "output_jsonl_sha256": _sha256(jsonl_p) if jsonl_p.exists() else None,
        "output_columns": OUTPUT_COLUMNS,
    }
