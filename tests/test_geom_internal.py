"""Internal-coordinate extraction and editing tests."""

import numpy as np
import pytest

from automol import Geometry, geom


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
    """Peroxide geometry fixture (H-O-O-H chain)."""
    return Geometry(
        symbols=["H", "O", "O", "H"],
        coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        charge=0,
        spin=0,
    )


def test__bonds(water: Geometry) -> None:
    """Test bond distance extraction for water."""
    amat = geom.adjacency_matrix(water)
    b = geom.bonds(water, amat)
    assert b.shape == (2, 3)
    assert np.array_equal(b, [[1.0, 8.0, 1.0], [1.0, 8.0, 1.0]])


def test__angles(water: Geometry) -> None:
    """Test bond angle extraction for water (exact 90 degree H-O-H angle)."""
    amat = geom.adjacency_matrix(water)
    a = geom.angles(water, amat)
    assert a.shape == (1, 4)
    assert np.array_equal(a[0, :3], [1.0, 8.0, 1.0])
    assert np.isclose(a[0, 3], np.pi / 2)


def test__dihedrals(peroxide: Geometry) -> None:
    """Test dihedral extraction for an H-O-O-H chain."""
    amat = geom.adjacency_matrix(peroxide)
    d = geom.dihedrals(peroxide, amat)
    assert d.shape == (1, 5)
    assert np.array_equal(d[0, :4], [0.0, 1.0, 2.0, 3.0])
    assert np.isclose(d[0, 4], np.pi / 2)


def test__set_distance(water: Geometry) -> None:
    """Test that set_distance updates the interatomic distance."""
    updated = geom.set_bond(water, idxs=[0, 1], val=1.2)
    dist = np.linalg.norm(updated.coordinates[1] - updated.coordinates[0])
    assert np.isclose(dist, 1.2)


def test__set_distance_wrong_index_count_raises(water: Geometry) -> None:
    """Test that set_distance rejects an index list that isn't length 2."""
    with pytest.raises(ValueError, match="Wrong number of indices"):
        geom.set_bond(water, idxs=[0, 1, 2], val=1.2)


def test__set_distance_exceeds_max_change_raises(water: Geometry) -> None:
    """Test that set_distance rejects a change beyond max_change."""
    with pytest.raises(ValueError, match="exceeds"):
        geom.set_bond(water, idxs=[0, 1], val=5.0, max_change=0.25)


def test__set_distance_in_place(water: Geometry) -> None:
    """Test that set_distance(in_place=True) mutates and returns the same object."""
    result = geom.set_bond(water, idxs=[0, 1], val=1.2, in_place=True)
    assert result is water
    dist = np.linalg.norm(water.coordinates[1] - water.coordinates[0])
    assert np.isclose(dist, 1.2)


def test__set_distance_not_in_place(water: Geometry) -> None:
    """Test that set_distance (default) does not mutate the original geometry."""
    original = water.coordinates.copy()
    geom.set_bond(water, idxs=[0, 1], val=1.2)
    assert np.allclose(water.coordinates, original)
