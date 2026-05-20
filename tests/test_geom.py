"""Geometry tests."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from automol import Geometry, geom


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # ty:ignore[invalid-argument-type]
    )


@pytest.fixture
def water_inv() -> Geometry:
    """Inverted Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[1, 0, 0], [0, 0, 0], [0, 1, 0]],  # ty:ignore[invalid-argument-type]
    )


@pytest.fixture
def peroxide() -> Geometry:
    """Peroxide geometry fixture."""
    return Geometry(
        symbols=["H", "O", "O", "H"],
        coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],  # ty:ignore[invalid-argument-type]
    )


def test__hash(water: Geometry) -> None:
    """Test geometry hashing."""
    water2 = Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1.000000000000001, 0]],  # ty:ignore[invalid-argument-type]
    )
    assert geom.geometry_hash(water) == geom.geometry_hash(water2)


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(geom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])


def test__inchi(water: Geometry) -> None:
    """Test InChI generation."""
    assert geom.inchi(water) == "InChI=1S/H2O/h1H2"


def test__is_similar(water: Geometry, water_inv: Geometry) -> None:
    """Test similarity analysis."""
    assert geom.is_similar(water, water)
    with pytest.raises(NotImplementedError):
        assert geom.is_similar(water, water_inv)


def test__read_xyz_file(water: Geometry, tmp_path: Path) -> None:
    """Test reading from xyz file."""
    xyz_path = tmp_path / "water.xyz"
    geom.write_xyz_file(water, xyz_path)
    water_out = geom.read_xyz_file(xyz_path)
    assert np.allclose(water.coordinates, water_out.coordinates)


def test__distance_matrix(water: Geometry) -> None:
    """Test distance matrix calculation."""
    dist_mat = geom.distance_matrix(water)
    expected = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
    assert np.allclose(dist_mat, expected)


def test__dihedral_angle(peroxide: Geometry) -> None:
    """Test dihedral angle calculation."""
    angle = geom.dihedral_angle(peroxide, (0, 1, 2, 3))
    expected = 90.0
    assert np.isclose(angle, expected)

    with pytest.raises(ValueError):  # noqa: PT011
        geom.dihedral_angle(peroxide, (0, 1, 2))


def test__reflection(peroxide: Geometry) -> None:
    """Test reflection."""
    normal = np.random.rand(3)  # noqa: NPY002
    refl_peroxide = geom.reflect(peroxide, normal)
    double_refl_peroxide = geom.reflect(refl_peroxide, normal)
    assert not np.allclose(peroxide.coordinates, refl_peroxide.coordinates)
    assert np.allclose(peroxide.coordinates, double_refl_peroxide.coordinates)


def test__to_eckart_frame(water: Geometry) -> None:
    """Test transformation to Eckart frame."""
    rot_water = geom.rotate(water, Rotation.random())

    align_water = geom.to_eckart_frame(water)
    align_rot_water = geom.to_eckart_frame(rot_water)

    assert align_water is not None
    assert align_rot_water is not None

    # Test currently not working
    # Need to canonicalize inertial axes
    # assert np.allclose(align_water.coordinates, align_rot_water.coordinates)  # noqa: ERA001, E501


def test__concat(water: Geometry) -> None:
    """Test geometry concatenation."""
    geo1 = Geometry(
        symbols=["O", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0]],  # ty:ignore[invalid-argument-type]
    )
    geo2 = Geometry(
        symbols=["H"],
        coordinates=[[0, 1, 0]],  # ty:ignore[invalid-argument-type]
    )
    concat_geo = geom.concat([geo1, geo2])
    assert water.symbols == concat_geo.symbols
    assert np.allclose(water.coordinates, concat_geo.coordinates)


def test__adjacency_matrix(water: Geometry) -> None:
    """Test adjacency matrix."""
    amat = geom.adjacency_matrix(geo=water)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])
