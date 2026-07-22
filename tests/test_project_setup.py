"""
Stage 0 — Project setup verification tests.

These tests confirm that the basic project structure is correct and that
the issue_intelligence package can be imported successfully.

Run with:
    pytest tests/ -v
"""

import issue_intelligence


def test_package_importable() -> None:
    """The issue_intelligence package must be importable from the venv."""
    assert issue_intelligence is not None


def test_package_version() -> None:
    """The package must expose a __version__ string matching 0.1.0."""
    assert issue_intelligence.__version__ == "0.1.0"
