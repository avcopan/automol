"""Geometry tests."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from automol import Geometry, geom, geoms
from automol.geom.canon import _truncate_bonds
from automol.graph import Atom, Bond, MolGraph

DATA_DIR = Path(__file__).parent / "data"

rng = np.random.default_rng(seed=679)

CARBON_VALENCY = 4


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
    assert geom.geometry_hash(water) == water.hash


def test__deterministic_hash(water: Geometry) -> None:
    """Test deterministic geometry hashing to the 5th decimal point."""
    water2 = water.model_copy(deep=True)
    water2.coordinates += 1e-5 * rng.uniform(low=-1, high=1, size=(water.atom_count, 3))
    assert geom.geometry_hash(water) == geom.geometry_hash(water2)


def test__deterministic_canonical_frame(propyl_oxirane: Geometry) -> None:
    """Test deterministic canonical frame."""
    ref = geom.canonical_frame(propyl_oxirane)

    translated = geom.transform.translate(
        propyl_oxirane, rng.uniform(low=-10, high=10, size=3)
    )
    rotated = geom.transform.rotate(translated, Rotation.random())
    trans_geo = geom.canonical_frame(rotated)

    assert geom.geometry_hash(trans_geo) == geom.geometry_hash(ref)


def test__determinisitic_canonical_order(propyl_oxirane: Geometry) -> None:
    """Test deterministic canonicalization of atom ordering."""
    perm1 = rng.permutation(propyl_oxirane.atom_count).tolist()
    propyl_oxirane.symbols = [propyl_oxirane.symbols[i] for i in perm1]
    propyl_oxirane.coordinates = propyl_oxirane.coordinates[perm1]
    canon1 = propyl_oxirane.canonical_form()

    perm1 = rng.permutation(propyl_oxirane.atom_count).tolist()
    propyl_oxirane.symbols = [propyl_oxirane.symbols[i] for i in perm1]
    propyl_oxirane.coordinates = propyl_oxirane.coordinates[perm1]
    assert propyl_oxirane.symbols != canon1.symbols
    canon2 = propyl_oxirane.canonical_form()

    assert canon1.hash == canon2.hash


def test__rdkit_roundtrip(water: Geometry) -> None:
    """Test Geometry to mol roundtrip."""
    mol = geom.rdkit_mol(water)
    geo_rt = geom.from_rdkit_mol(mol)

    assert water.hash == geo_rt.hash


def test__xyz_roundtrip(water: Geometry) -> None:
    """Test Geometry to xyz string roundtrip."""
    xyz = water.xyz_block()
    geo_rt = geom.from_xyz_block(xyz, charge=0, spin=0)

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
    refl_peroxide = geom.transform.reflect(peroxide, normal)
    double_refl_peroxide = geom.transform.reflect(refl_peroxide, normal)
    assert not np.allclose(peroxide.coordinates, refl_peroxide.coordinates)
    assert np.allclose(peroxide.coordinates, double_refl_peroxide.coordinates)


def test__to_eckart_frame(water: Geometry) -> None:
    """Test transformation to Eckart frame."""
    rot_water = geom.transform.rotate(water, Rotation.random())

    align_water = geom.transform.eckart_frame(water)
    align_rot_water = geom.transform.eckart_frame(rot_water)

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

    concat_geo = geoms.concat([geo1, geo2])
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
    freqs, _ = geom.properties.vibrational_analysis(
        propyl_oxirane,
        propyl_oxirane_hessian,  # ty:ignore[invalid-argument-type]
    )

    assert len(freqs) == len(orca_frequencies_propyl_oxirane[6:])
    assert np.allclose(freqs, orca_frequencies_propyl_oxirane[6:], rtol=0.05)


def _make_graph(
    symbols: list[str], coords: list[list[float]], bonds: dict[tuple[int, int], float]
) -> MolGraph:
    """Build a MolGraph with given atoms and bond distances."""
    gra = MolGraph(atom_type=Atom, bond_type=Bond)

    for i, (sym, c) in enumerate(zip(symbols, coords, strict=True)):
        atom = Atom(symbol=sym, coords=np.array(c))
        gra.add_node(i, **atom.model_dump())

    for (u, v), dist in bonds.items():
        gra.add_edge(u, v, **Bond(distance=dist).model_dump())

    return gra


def test__truncate_bonds_removes_worst_overstretched_bond() -> None:
    """Test that a hypervalent atoms most-stretched bond is removed."""
    # Carbon at origin, bonded to 5 hydrogens.
    symbols = ["C", "H", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    # Bond 0-5 is deliberately the longest -> should be the one truncated.
    bonds = {
        (0, 1): 1.09,
        (0, 2): 1.09,
        (0, 3): 1.09,
        (0, 4): 1.09,
        (0, 5): 1.50,  # most stretched relative to covalent radii
    }
    gra = _make_graph(symbols, coords, bonds)

    assert gra.degree(0) > CARBON_VALENCY  # hypervalent before truncation

    truncated = _truncate_bonds(gra)

    assert truncated.degree(0) == CARBON_VALENCY
    assert not truncated.has_edge(0, 5)
    # All remaining bonds should still be intact
    for u, v in [(0, 1), (0, 2), (0, 3), (0, 4)]:
        assert truncated.has_edge(u, v)


def test__truncate_bonds_no_hypervalency_is_noop() -> None:
    """Test that a graph with no hypervalent atoms is returned unchanged."""
    symbols = ["C", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    bonds = {
        (0, 1): 1.09,
        (0, 2): 1.09,
        (0, 3): 1.09,
        (0, 4): 1.09,
    }
    gra = _make_graph(symbols, coords, bonds)

    truncated = _truncate_bonds(gra)

    assert set(truncated.edges()) == set(gra.edges())
    assert truncated.degree(0) == CARBON_VALENCY


def test__truncate_bonds_raises_without_distance() -> None:
    """Test that a hypervalent atom with missing bond distance info is raised."""
    symbols = ["C", "H", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    gra = MolGraph(atom_type=Atom, bond_type=Bond)
    for i, (sym, c) in enumerate(zip(symbols, coords, strict=True)):
        atom = Atom(symbol=sym, coords=np.array(c))
        gra.add_node(i, **atom.model_dump())

    # All bonds missing distance (None, the default)
    for u, v in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]:
        gra.add_edge(u, v, **Bond().model_dump())

    with pytest.raises(ValueError, match="Cannot manage hypervalencies"):
        _truncate_bonds(gra)
