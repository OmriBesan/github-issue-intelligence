"""
issue_intelligence.data.combine
================================
Utilities for combining, deduplicating, and analysing multiple raw issue
datasets collected from the GitHub API.

All functions operate on plain Python structures (lists of dicts) so they
remain testable without file I/O.  File-level helpers wrap these functions.

Public API
----------
load_and_combine(paths, sort_by_date=True)
    Load multiple JSON issue files, deduplicate by GitHub issue ID, and
    return the merged list together with combination statistics.

find_yearly_gaps(issues, min_year=None, max_year=None)
    Identify calendar years with no issues (coverage gaps).

sample_issues_by_label(issues, label, n=40, seed=42)
    Randomly sample up to n issues that carry a specific raw label name.
    Sampling is deterministic for a given seed.

build_cooccurrence_stats(issues, target_labels, reference_labels)
    Count how often each target label co-occurs with each reference label.

save_combined(issues, metadata, output_path)
    Save the combined issue list and metadata JSON to disk.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

# Make the src package importable when this module is used directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from issue_intelligence.data.audit import compute_yearly_distribution, load_issues

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_combine(
    paths: list[Path | str],
    sort_by_date: bool = True,
) -> tuple[list[dict], dict]:
    """
    Load multiple raw issue JSON files and deduplicate by GitHub issue ID.

    Parameters
    ----------
    paths : list of Path or str
        Ordered list of input JSON files.  Files are processed in order;
        issues already seen in an earlier file are silently skipped in
        later files.
    sort_by_date : bool
        If True (default), the returned list is sorted by ``created_at``
        ascending.  This ensures consistent ordering regardless of which
        file was processed first.

    Returns
    -------
    issues : list of dict
        Deduplicated issue records.
    stats : dict
        Combination statistics including ``total_issues``,
        ``duplicates_found``, and ``issues_per_file``.
    """
    seen_ids: set[int] = set()
    all_issues: list[dict] = []
    duplicates_found = 0
    issues_per_file: dict[str, int] = {}

    for path in paths:
        p = Path(path)
        data = load_issues(p)
        new_count = 0
        for issue in data:
            issue_id = issue.get("id")
            if issue_id in seen_ids:
                duplicates_found += 1
                continue
            if issue_id is not None:
                seen_ids.add(issue_id)
            all_issues.append(issue)
            new_count += 1
        issues_per_file[str(p)] = new_count

    if sort_by_date:
        all_issues.sort(key=lambda x: x.get("created_at") or "")

    return all_issues, {
        "files_loaded": len(paths),
        "total_issues": len(all_issues),
        "duplicates_found": duplicates_found,
        "issues_per_file": issues_per_file,
    }


def find_yearly_gaps(
    issues: list[dict],
    min_year: int | None = None,
    max_year: int | None = None,
) -> dict:
    """
    Find calendar years within the dataset's span that have no issues.

    Parameters
    ----------
    issues : list of dict
        Issue records.  Each record must have a ``created_at`` field.
    min_year : int, optional
        Start of the expected coverage window.  Defaults to the earliest
        year found in the data.
    max_year : int, optional
        End of the expected coverage window.  Defaults to the latest year
        found in the data.

    Returns
    -------
    dict with keys:
        ``years_present`` — sorted list of year strings that have data.
        ``gaps`` — sorted list of year strings with zero issues.
        ``yearly_counts`` — {year_str: count} for all years with data.
        ``min_year`` — str, first year of the checked window.
        ``max_year`` — str, last year of the checked window.
    """
    yearly = compute_yearly_distribution(issues)
    years_with_data = sorted(yearly.keys())

    if not years_with_data:
        return {
            "years_present": [],
            "gaps": [],
            "yearly_counts": {},
            "min_year": None,
            "max_year": None,
        }

    effective_min = min_year or int(years_with_data[0])
    effective_max = max_year or int(years_with_data[-1])

    gaps = [
        str(y)
        for y in range(effective_min, effective_max + 1)
        if yearly.get(str(y), 0) == 0
    ]

    return {
        "years_present": years_with_data,
        "gaps": gaps,
        "yearly_counts": dict(yearly),
        "min_year": str(effective_min),
        "max_year": str(effective_max),
    }


def sample_issues_by_label(
    issues: list[dict],
    label: str,
    n: int = 40,
    seed: int = 42,
) -> list[dict]:
    """
    Randomly sample up to *n* issues that carry *label* as a raw label name.

    The sample is deterministic: the same ``seed`` always produces the same
    result for the same input list.

    Parameters
    ----------
    issues : list of dict
        Issue records.  Each record must have a ``labels`` field containing
        a list of label name strings.
    label : str
        Exact raw label name to filter on.
    n : int
        Maximum sample size (default: 40).
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    list of dict
        Sampled issue records (at most n, possibly fewer).
    """
    rng = random.Random(seed)
    candidates = [i for i in issues if label in (i.get("labels") or [])]
    k = min(n, len(candidates))
    return rng.sample(candidates, k)


def build_cooccurrence_stats(
    issues: list[dict],
    target_labels: list[str],
    reference_labels: list[str],
) -> dict:
    """
    Count how often each target label co-occurs with each reference label.

    Only issues that carry a target label are included in the analysis.
    An issue is counted once per (target, reference) pair.

    Parameters
    ----------
    issues : list of dict
        Issue records.
    target_labels : list of str
        Labels of interest (e.g. ``["Enhancement", "New Feature", "RFC"]``).
    reference_labels : list of str
        Labels to measure co-occurrence against
        (e.g. ``["Bug", "Documentation"]``).

    Returns
    -------
    dict
        Nested dict ``{target_label: {reference_label: count}}``.
        Also includes ``issues_with_target`` and ``total_issues_checked``.
    """
    cooc: dict[str, Counter] = {t: Counter() for t in target_labels}
    issues_with_target: dict[str, int] = {t: 0 for t in target_labels}

    for issue in issues:
        lbls = set(issue.get("labels") or [])
        for target in target_labels:
            if target in lbls:
                issues_with_target[target] += 1
                for ref in reference_labels:
                    if ref in lbls:
                        cooc[target][ref] += 1

    return {
        "cooccurrence": {t: dict(c) for t, c in cooc.items()},
        "issues_with_target": issues_with_target,
        "total_issues_checked": len(issues),
    }


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


def save_combined(
    issues: list[dict],
    metadata: dict,
    output_path: Path | str,
) -> None:
    """
    Save the combined issue list and metadata JSON to disk.

    The metadata file is written beside the issue file with ``_metadata``
    appended to the stem.

    Parameters
    ----------
    issues : list of dict
    metadata : dict
    output_path : Path or str
        Path to the issues JSON file (e.g.
        ``data/raw/scikit-learn_issues_combined.json``).
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = p.with_name(p.stem + "_metadata" + p.suffix)
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
