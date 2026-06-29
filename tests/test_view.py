"""View tests."""

import pytest

from automol import Geometry, View


@pytest.fixture
def view() -> View:
    """Empty view for testing."""
    return View()


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # ty:ignore[invalid-argument-type]
        charge=0,
        spin=0,
    )


def test__add_geometry(view: View, water: Geometry) -> None:
    """Test add geometry."""
    view.add_geometry(water, label=True)


def test__add_axes(view: View) -> None:
    """Test add axes."""
    view.add_xyz_axes(scale=2)
