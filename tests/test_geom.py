"""Geometry tests: model, I/O, rdkit conversion, analysis, and transforms."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from rdkit import Chem
from scipy.spatial.transform import Rotation

from automol import Geometry, geom, rd
from automol.utils.exc import XYZFormatError

# Model


def test__masses_atomic_numbers_valences(water: Geometry) -> None:
    """Test scalar per-atom properties."""
    expected_masses = [
        pytest.approx(15.9949, abs=1e-4),
        pytest.approx(1.00783, abs=1e-4),
        pytest.approx(1.00783, abs=1e-4),
    ]
    assert water.masses == expected_masses
    assert water.atomic_numbers == [8, 1, 1]
    assert water.valences == [6, 1, 1]


def test__coordinates_symbols_mismatch_raises() -> None:
    """Test that mismatched symbols/coordinates lengths are rejected."""
    with pytest.raises(ValueError, match="does not match"):
        Geometry(
            symbols=["O", "H", "H"],
            coordinates=[[0, 0, 0], [1, 0, 0]],
            charge=0,
            spin=0,
        )


def test__symbols_canonicalized() -> None:
    """Test that symbols are validated case-insensitively and canonicalized."""
    geo = Geometry(symbols=["cl", "CL"], coordinates=np.eye(2, 3), charge=0, spin=0)
    assert geo.symbols == ["Cl", "Cl"]


def test__unknown_symbol_raises() -> None:
    """Test that unknown element symbols are rejected."""
    with pytest.raises(ValidationError, match="No element"):
        Geometry(symbols=["Xx"], coordinates=[[0, 0, 0]], charge=0, spin=0)


@pytest.mark.parametrize(("charge", "spin"), [(0, 1), (0, -2), (1, 0), (11, 0)])
def test__inconsistent_spin_raises(charge: int, spin: int) -> None:
    """Test that spin inconsistent with the electron count is rejected."""
    with pytest.raises(ValidationError, match="inconsistent"):
        Geometry(
            symbols=["O", "H", "H"], coordinates=np.eye(3), charge=charge, spin=spin
        )


def test__invalid_assignment_is_rejected(water: Geometry) -> None:
    """Test that invalid assignments raise and leave the geometry unchanged."""
    with pytest.raises(ValidationError, match="does not match"):
        water.symbols = ["O", "H"]
    with pytest.raises(ValidationError, match="inconsistent"):
        water.spin = 1
    assert water.symbols == ["O", "H", "H"]
    assert water.spin == 0


def test__equality(water: Geometry, peroxide: Geometry) -> None:
    """Test that geometries compare by value, including coordinates."""
    assert water == water.model_copy(deep=True)
    assert water != peroxide
    assert water != geom.translate(water, [1e-8, 0, 0])
    assert water != "water"


def test__relabel_atoms(water: Geometry) -> None:
    """Test atom reordering."""
    relabeled = water.relabel_atoms([1, 0, 2])
    assert relabeled.symbols == ["H", "O", "H"]
    assert np.array_equal(relabeled.coordinates, water.coordinates[[1, 0, 2]])


@pytest.mark.parametrize("indices", [[0, 0, 0], [0, 1], [0, 1, 3]])
def test__relabel_atoms_non_permutation_raises(
    water: Geometry, indices: list[int]
) -> None:
    """Test that non-permutation indices are rejected."""
    with pytest.raises(ValueError, match="not a permutation"):
        water.relabel_atoms(indices)


# XYZ I/O


def test__xyz_file_roundtrip(water: Geometry, tmp_path: Path) -> None:
    """Test xyz file roundtrip through both the method and module-level API."""
    method_path = tmp_path / "method.xyz"
    water.xyz_file(path=method_path)
    function_path = tmp_path / "function.xyz"
    geom.xyz_file(water, path=function_path)

    for geo_rt in (
        Geometry.from_xyz_file(method_path, charge=0, spin=0),
        geom.from_xyz_file(function_path, charge=0, spin=0),
    ):
        assert water.symbols == geo_rt.symbols
        assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__xyz_file_trailing_newline(water: Geometry, tmp_path: Path) -> None:
    """Test that xyz files end with a newline."""
    path = tmp_path / "water.xyz"
    water.xyz_file(path=path)
    assert path.read_text().endswith("\n")
    assert not path.read_text().endswith("\n\n")


def test__xyz_block_multiline_comment_raises(water: Geometry) -> None:
    """Test that a multi-line comment is rejected."""
    with pytest.raises(ValueError, match="single line"):
        water.xyz_block(comment="line 1\nline 2")


def test__from_xyz_block_flexible_formats() -> None:
    """Test leading-dot/exponent floats, lowercase symbols, and atomic numbers."""
    xyz = "3\n\no .5 -.5 1e-1\n1 0 0 0\nH\t+1.0E+0 2 3.  \n"
    geo = geom.from_xyz_block(xyz, charge=0, spin=0)
    assert geo.symbols == ["O", "H", "H"]
    assert np.allclose(geo.coordinates, [[0.5, -0.5, 0.1], [0, 0, 0], [1, 2, 3]])


@pytest.mark.parametrize(
    ("xyz_block", "match"),
    [
        ("", "empty"),
        ("1\n\nnot a valid xyz line\n", "Failed to parse"),
        ("1\n\nH1 0 0 0\n", "Failed to parse"),
        ("H 0 0 0\n", "atom count"),
        ("1\ncomment\n", "no atoms"),
        ("3\n\nH 0 0 0\nH 0 0 1\n", "does not match"),
        ("1\na\nH 0 0 0\n1\nb\nH 0 0 1\n", "Multi-frame"),
        ("1\n\n999 0 0 0\n", "Unknown element"),
        ("1\n\nXx 0 0 0\n", "Unknown element"),
    ],
)
def test__from_xyz_block_raises(xyz_block: str, match: str) -> None:
    """Test that empty and malformed xyz blocks are rejected."""
    with pytest.raises(XYZFormatError, match=match):
        geom.from_xyz_block(xyz_block, charge=0, spin=0)


# RDKit conversion (C/H/O perception is covered in test_geom_to_mol.py)


@pytest.mark.parametrize(
    "smiles",
    [
        "C[N+](=O)[O-]",
        "[NH4+]",
        "CS(C)=O",
        "O=[N][O]",
        "[NH3+]CC(=O)[O-]",
        "F[C@H](Cl)Br",
    ],
)
def test__rdkit_mol_perceives_structure(smiles: str) -> None:
    """Test that heteroatom bond orders, charges, radicals, and stereo roundtrip."""
    geo = geom.from_rdkit_mol(rd.mol.from_smiles(smiles, with_coords=True))
    mol = geom.rdkit_mol(geo)

    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == expected
    assert rd.mol.charge(mol) == geo.charge
    assert rd.mol.spin(mol) == geo.spin


def test__from_rdkit_mol_without_coordinates() -> None:
    """Test that Geometry is built from an rdkit Mol missing coordinates."""
    mol = rd.mol.from_smiles("O", with_coords=False)
    assert not rd.mol.has_coordinates(mol)

    geo = geom.from_rdkit_mol(mol)
    assert sorted(geo.symbols) == ["H", "H", "O"]
    assert geo.coordinates.shape == (3, 3)


def test__from_rdkit_mol_with_coordinates() -> None:
    """Test that Geometry is built from an rdkit Mol that already has coordinates."""
    mol = rd.mol.from_smiles("O", with_coords=True)
    assert rd.mol.has_coordinates(mol)
    original_coords = rd.mol.coordinates(mol)

    geo = geom.from_rdkit_mol(mol)
    assert sorted(geo.symbols) == ["H", "H", "O"]
    sorted_geo_coords = np.sort(geo.coordinates, axis=0)
    sorted_original_coords = np.sort(original_coords, axis=0)
    assert np.allclose(sorted_geo_coords, sorted_original_coords)


# Analysis


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(geom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])


def test__empty_geometry_properties() -> None:
    """Test mass properties of an empty geometry."""
    empty = Geometry(symbols=[], coordinates=np.zeros((0, 3)), charge=0, spin=0)
    assert np.array_equal(geom.center_of_mass(empty), np.zeros(3))
    assert np.array_equal(geom.inertia_tensor(empty), np.zeros((3, 3)))


def test__inertia_tensor_values() -> None:
    """Test the inertia tensor of a linear molecule along z."""
    h2 = Geometry(
        symbols=["H", "H"], coordinates=[[0, 0, -1], [0, 0, 1]], charge=0, spin=0
    )
    mass = h2.masses[0]
    expected = np.diag([2 * mass, 2 * mass, 0.0])
    assert np.allclose(geom.inertia_tensor(h2), expected)


def test__inertia_tensor(water: Geometry) -> None:
    """Test that the inertia tensor is a symmetric (3, 3) matrix."""
    tensor = geom.inertia_tensor(water)
    assert tensor.shape == (3, 3)
    assert np.allclose(tensor, tensor.T)


def test__distance_matrix(water: Geometry) -> None:
    """Test distance matrix calculation."""
    dist_mat = geom.distance_matrix(water)
    expected = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
    assert np.allclose(dist_mat, expected)


def test__adjacency_matrix(water: Geometry) -> None:
    """Test adjacency matrix."""
    amat = geom.adjacency_matrix(geo=water)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])


def test__adjacency_matrix_flood_fill() -> None:
    """Test that flood fill connects otherwise disconnected fragments."""
    disconnected = Geometry(
        symbols=["H", "H"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    amat = geom.adjacency_matrix(geo=disconnected, flood_fill=True)
    assert np.array_equal(amat, [[0, 1], [1, 0]])


def test__adjacency_matrix_flood_fill_unconnectable() -> None:
    """Test that flood fill terminates when valence caps prevent connection."""
    two_h2 = Geometry(
        symbols=["H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74], [0, 0, 5], [0, 0, 5.74]],
        charge=0,
        spin=0,
    )
    amat = geom.adjacency_matrix(two_h2, flood_fill=True, enforce_valence=True)
    expected = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    assert np.array_equal(amat, expected)


def test__adjacency_matrix_enforce_valence_order_independent() -> None:
    """Test that valence enforcement keeps the closest bonds for any atom order."""
    # H3 chain with a short and a long H-H contact: only the short bond survives
    h3 = Geometry(
        symbols=["H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74], [0, 0, 1.6], [0, 0, 2.34]],
        charge=0,
        spin=0,
    )
    expected = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    for perm in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]):
        geo = h3.relabel_atoms(perm)
        amat = geom.adjacency_matrix(geo, enforce_valence=True)
        inv = np.argsort(perm)
        assert np.array_equal(amat[np.ix_(inv, inv)], expected)


def test__distance_keys(water: Geometry) -> None:
    """Test distance descriptor generation."""
    keys = geom.distance_keys(water)
    assert keys.shape == (3, 3)
    # Pairs must be sorted (z1 <= z2), then by z1, z2, distance.
    assert np.all(keys[:, 0] <= keys[:, 1])
    assert np.array_equal(keys[:, :2], np.sort(keys[:, :2], axis=1))


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
    assert np.array_equal(d[0, :4], [1.0, 8.0, 8.0, 1.0])
    assert np.isclose(d[0, 4], np.pi / 2)


def test__internal_coordinates_counts() -> None:
    """Test bond/angle/dihedral counts for propane, including the empty case."""
    propane = geom.from_rdkit_mol(rd.mol.from_smiles("CCC", with_coords=True))
    amat = geom.adjacency_matrix(propane)
    assert geom.bonds(propane, amat).shape == (10, 3)
    # Angles: 6 at each of the three carbons (4 neighbors each, choose 2)
    assert geom.angles(propane, amat).shape == (18, 4)
    # Dihedrals: 3 x 3 = 9 torsions about each of the two C-C bonds
    assert geom.dihedrals(propane, amat).shape == (18, 5)

    he = Geometry(symbols=["He"], coordinates=[[0, 0, 0]], charge=0, spin=0)
    he_amat = geom.adjacency_matrix(he)
    assert geom.bonds(he, he_amat).shape == (0, 3)
    assert geom.angles(he, he_amat).shape == (0, 4)
    assert geom.dihedrals(he, he_amat).shape == (0, 5)


def test__dihedrals_orientation_invariant(peroxide: Geometry) -> None:
    """Test that reversing atom order does not change dihedral rows."""
    amat = geom.adjacency_matrix(peroxide)
    reversed_peroxide = peroxide.relabel_atoms([3, 2, 1, 0])
    reversed_amat = geom.adjacency_matrix(reversed_peroxide)
    assert np.allclose(
        geom.dihedrals(peroxide, amat),
        geom.dihedrals(reversed_peroxide, reversed_amat),
    )


def test__internal_coordinates_bad_amat_raises(water: Geometry) -> None:
    """Test that a wrongly shaped adjacency matrix is rejected."""
    with pytest.raises(ValueError, match="shape"):
        geom.bonds(water, np.zeros((2, 2)))


# Transforms


def test__reflection(peroxide: Geometry) -> None:
    """Test reflection."""
    normal = np.random.rand(3)  # noqa: NPY002
    refl_peroxide = geom.core.reflect(peroxide, normal)
    double_refl_peroxide = geom.core.reflect(refl_peroxide, normal)
    assert not np.allclose(peroxide.coordinates, refl_peroxide.coordinates)
    assert np.allclose(
        peroxide.coordinates, double_refl_peroxide.coordinates, atol=1e-7
    )


def test__reflection_zero_normal_raises(peroxide: Geometry) -> None:
    """Test that a zero normal vector is rejected."""
    with pytest.raises(ValueError, match="non-zero"):
        geom.reflect(peroxide, [0, 0, 0])


def test__translate(water: Geometry) -> None:
    """Test translation."""
    shift = [1.0, 2.0, 3.0]
    translated = geom.core.translate(water, shift)
    assert np.allclose(translated.coordinates, water.coordinates + shift)
    assert not np.allclose(translated.coordinates, water.coordinates)


def test__translate_in_place(water: Geometry) -> None:
    """Test in-place translation."""
    original = water.coordinates.copy()
    result = geom.core.translate(water, [1.0, 0.0, 0.0], in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, np.add(original, [1.0, 0.0, 0.0]))


def test__translate_with_keys(water: Geometry) -> None:
    """Test translation of a subset of atoms."""
    original = water.coordinates.copy()
    translated = geom.core.translate(water, [1.0, 0.0, 0.0], keys=[0])
    assert np.allclose(translated.coordinates[0], original[0] + [1.0, 0.0, 0.0])
    assert np.allclose(translated.coordinates[1:], original[1:])


def test__rotate(water: Geometry) -> None:
    """Test rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    rotated = geom.core.rotate(water, rot)
    assert np.allclose(rotated.coordinates, rot.apply(water.coordinates))


def test__rotate_in_place(water: Geometry) -> None:
    """Test in-place rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    expected = rot.apply(water.coordinates)
    result = geom.core.rotate(water, rot, in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, expected)
