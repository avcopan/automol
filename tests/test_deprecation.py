"""Deprecation tests."""

import importlib

import pytest

import automol


def test_import_warns() -> None:
    """Importing automol warns that it is deprecated in favor of autogeom."""
    with pytest.warns(FutureWarning, match="autogeom"):
        importlib.reload(automol)
