"""Shared fixtures for geometry tests."""

import pytest

from automol import Geometry


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )


@pytest.fixture
def peroxide() -> Geometry:
    """Peroxide geometry fixture."""
    return Geometry(
        symbols=["H", "O", "O", "H"],
        coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        charge=0,
        spin=0,
    )
