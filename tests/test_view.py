"""View tests."""

from pathlib import Path

import pytest

from automol import Geometry, View
from automol.geom.io import render_gif, render_svg
from automol.geom.io import view as build_view


@pytest.fixture
def view() -> View:
    """Empty view for testing."""
    return View()


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )


def test__add_geometry(view: View, water: Geometry) -> None:
    """Test add geometry."""
    view.add_geometry(water, label=True)


def test__add_axes(view: View) -> None:
    """Test add axes."""
    view.add_xyz_axes(scale=2)


def test__add_vectors_mismatched_colors_raises(view: View) -> None:
    """Test that mismatched coords/colors lengths are rejected."""
    with pytest.raises(ValueError, match="do not match"):
        view.add_vectors([[1, 0, 0], [0, 1, 0]], colors=["red"])


def test__add_vector_direction(view: View) -> None:
    """Test adding a vector as a direction from a start coordinate."""
    view.add_vector([1, 0, 0], start_coord=[1, 1, 1], direction=True)


def test__view_function(water: Geometry) -> None:
    """Test py3Dmol view construction."""
    result = build_view(water, label=True)
    assert result is not None


def test__view_function_without_label(water: Geometry) -> None:
    """Test py3Dmol view construction without atom labels."""
    result = build_view(water, label=False)
    assert result is not None


def test__render_svg(water: Geometry, tmp_path: Path) -> None:
    """Test svg rendering."""
    out = tmp_path / "water"
    result = render_svg(water, out=out)
    assert (tmp_path / "water.svg").exists()
    assert "<svg" in str(result)


def test__render_gif(water: Geometry, tmp_path: Path) -> None:
    """Test gif rendering."""
    out = tmp_path / "water"
    render_gif(water, out=out)
    assert (tmp_path / "water.gif").exists()
