"""Geometry properties tests."""

import numpy as np

from automol import Geometry, geom


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(geom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])


def test__distance_matrix(water: Geometry) -> None:
    """Test distance matrix calculation."""
    dist_mat = geom.distance_matrix(water)
    expected = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
    assert np.allclose(dist_mat, expected)


def test__adjacency_matrix(water: Geometry) -> None:
    """Test adjacency matrix."""
    amat = geom.adjacency_matrix(geo=water)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])


def test__adjacency_matrix_flood_fill(water: Geometry) -> None:
    """Test adjacency matrix flood fill."""
    amat = geom.adjacency_matrix(geo=water, flood_fill=True)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    disconnected = Geometry(
        symbols=["H", "H"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    amat = geom.adjacency_matrix(geo=disconnected, flood_fill=True)
    assert np.array_equal(amat, [[0, 1], [1, 0]])


def test__distance_keys(water: Geometry) -> None:
    """Test distance descriptor generation."""
    keys = geom.distance_keys(water)
    assert keys.shape == (3, 3)
    # Pairs must be sorted (z1 <= z2), then by z1, z2, distance.
    assert np.all(keys[:, 0] <= keys[:, 1])
    assert np.array_equal(keys[:, :2], np.sort(keys[:, :2], axis=1))
