"""Smoke test: verify the src package is importable."""

import importlib


def test_src_importable() -> None:
    """src package can be imported without error."""
    mod = importlib.import_module("src")
    assert mod is not None
