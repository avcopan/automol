"""Geometry tests."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from automol import Geometry, geom
from automol.utils.exc import HashGenerationError

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # ty:ignore[invalid-argument-type]
        charge=0,
        spin=0,
    )


@pytest.fixture
def peroxide() -> Geometry:
    """Peroxide geometry fixture."""
    return Geometry(
        symbols=["H", "O", "O", "H"],
        coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],  # ty:ignore[invalid-argument-type]
        charge=0,
        spin=0,
    )


@pytest.fixture
def propyl_oxirane() -> Geometry:
    """Propyl oxirane geometry fixture."""
    return Geometry.from_xyz_file(DATA_DIR / "propyl_oxirane.xyz", charge=0, spin=1)


@pytest.fixture
def propyl_oxirane_hessian() -> list[list[float]]:
    """Propyl oxirane Hessian fixture."""
    return np.loadtxt(DATA_DIR / "propyl_oxirane_hessian.gz").tolist()


@pytest.fixture
def orca_frequencies_propyl_oxirane() -> list[float]:
    """Fixture for orca freqencies of propyl oxirane test data."""
    return np.loadtxt(DATA_DIR / "propyl_oxirane_frequencies.gz").tolist()


def test__hash(water: Geometry) -> None:
    """Test geometry hashing."""
    exp_hash = "67eecf41909c735495d035b556a1adf51f9fb9e1c5a6219be36a2b31a2dd3fa4"
    assert geom.geometry_hash(water) == exp_hash


def test__unhashable() -> None:
    """Test that geometry hashing raises exception without necessary fields."""
    geo = Geometry(
        symbols=["H"], coordinates=np.array([[0, 0, 0]]), charge=None, spin=None
    )
    with pytest.raises(HashGenerationError):
        geom.geometry_hash(geo)


def test__deterministic_hash(water: Geometry) -> None:
    """Test deterministic geometry hashing."""
    water2 = Geometry(
        symbols=["O", "H", "H"],
        coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1.000000000000001, 0]]),
        charge=0,
        spin=0,
    )
    assert geom.geometry_hash(water) == geom.geometry_hash(water2)


def test__rdkit_roundtrip(water: Geometry) -> None:
    """Test Geometry to mol roundtrip."""
    mol = water.rdkit_mol()
    geo_rt = Geometry.from_rdkit_mol(mol)

    assert water.hash == geo_rt.hash


def test__xyz_roundtrip(water: Geometry) -> None:
    """Test Geometry to xyz string roundtrip."""
    xyz = water.xyz_block()
    geo_rt = Geometry.from_xyz_block(xyz)

    assert water.symbols == geo_rt.symbols
    assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(geom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])


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
        charge=0,
        spin=0,
    )
    geo2 = Geometry(
        symbols=["H"],
        coordinates=[[0, 1, 0]],  # ty:ignore[invalid-argument-type]
        charge=0,
        spin=0,
    )

    concat_geo = geom.concat([geo1, geo2])
    assert water.symbols == concat_geo.symbols
    assert np.allclose(water.coordinates, concat_geo.coordinates)


def test__adjacency_matrix(water: Geometry) -> None:
    """Test adjacency matrix."""
    amat = geom.adjacency_matrix(geo=water)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])


def test__vibrational_analysis(
    propyl_oxirane: Geometry,
    propyl_oxirane_hessian: list[list[float]],
    orca_frequencies_propyl_oxirane: list[float],
) -> None:
    """Test vibrational analysis."""
    freqs, _ = geom.vibrational_analysis(propyl_oxirane, propyl_oxirane_hessian)

    assert len(freqs) == len(orca_frequencies_propyl_oxirane[6:])
    assert np.allclose(freqs, orca_frequencies_propyl_oxirane[6:], rtol=0.05)
