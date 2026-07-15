"""Geometry tests."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from automol import Geometry, geom, rd
from automol.utils.exc import XYZFormatError

DATA_DIR = Path(__file__).parent / "data"


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


def test__rdkit_roundtrip(water: Geometry) -> None:
    """Test Geometry to mol roundtrip."""
    mol = geom.rdkit_mol(water)
    geo_rt = geom.from_rdkit_mol(mol)

    assert geo_rt.symbols == water.symbols
    assert np.allclose(
        geom.distance_matrix(geo_rt), geom.distance_matrix(water), rtol=0.95
    )
    assert geo_rt.charge == water.charge
    assert geo_rt.spin == water.spin


def test__to_ase(water: Geometry) -> None:
    """Test Geometry to ASE Atoms conversion."""
    atoms = geom.to_ase(water)

    assert atoms.get_chemical_symbols() == water.symbols
    assert np.allclose(atoms.positions, water.coordinates)
    assert atoms.info["charge"] == water.charge
    assert atoms.info["spin"] == water.spin


def test__xyz_roundtrip(water: Geometry) -> None:
    """Test Geometry to xyz string roundtrip."""
    xyz = water.xyz_block()
    geo_rt = geom.from_xyz_block(xyz, charge=0, spin=0)

    assert water.symbols == geo_rt.symbols
    assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__hill_formula(water: Geometry) -> None:
    """Test Geometry to Hill-ordered formula."""
    assert geom.hill_formula(water) == "H2O"


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(geom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])


def test__distance_matrix(water: Geometry) -> None:
    """Test distance matrix calculation."""
    dist_mat = geom.distance_matrix(water)
    expected = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
    assert np.allclose(dist_mat, expected)


def test__reflection(peroxide: Geometry) -> None:
    """Test reflection."""
    normal = np.random.rand(3)  # noqa: NPY002
    refl_peroxide = geom.transform.reflect(peroxide, normal)
    double_refl_peroxide = geom.transform.reflect(refl_peroxide, normal)
    assert not np.allclose(peroxide.coordinates, refl_peroxide.coordinates)
    assert np.allclose(
        peroxide.coordinates, double_refl_peroxide.coordinates, atol=1e-7
    )


def test__adjacency_matrix(water: Geometry) -> None:
    """Test adjacency matrix."""
    amat = geom.adjacency_matrix(geo=water)
    assert np.array_equal(amat, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])


def test__adjacency_matrix_flood_fill_not_implemented(water: Geometry) -> None:
    """Test that flood_fill is not yet implemented."""
    with pytest.raises(NotImplementedError):
        geom.adjacency_matrix(geo=water, flood_fill=True)


def test__distance_keys(water: Geometry) -> None:
    """Test distance descriptor generation."""
    keys = geom.distance_keys(water)
    assert keys.shape == (3, 3)
    # Pairs must be sorted (z1 <= z2), then by z1, z2, distance.
    assert np.all(keys[:, 0] <= keys[:, 1])
    assert np.array_equal(keys[:, :2], np.sort(keys[:, :2], axis=1))


def test__translate(water: Geometry) -> None:
    """Test translation."""
    shift = [1.0, 2.0, 3.0]
    translated = geom.transform.translate(water, shift)
    assert np.allclose(translated.coordinates, water.coordinates + shift)
    assert not np.allclose(translated.coordinates, water.coordinates)


def test__translate_in_place(water: Geometry) -> None:
    """Test in-place translation."""
    original = water.coordinates.copy()
    result = geom.transform.translate(water, [1.0, 0.0, 0.0], in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, np.add(original, [1.0, 0.0, 0.0]))


def test__translate_with_keys(water: Geometry) -> None:
    """Test translation of a subset of atoms."""
    original = water.coordinates.copy()
    translated = geom.transform.translate(water, [1.0, 0.0, 0.0], keys=[0])
    assert np.allclose(translated.coordinates[0], original[0] + [1.0, 0.0, 0.0])
    assert np.allclose(translated.coordinates[1:], original[1:])


def test__rotate(water: Geometry) -> None:
    """Test rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    rotated = geom.transform.rotate(water, rot)
    assert np.allclose(rotated.coordinates, rot.apply(water.coordinates))
    dist_before = geom.distance_matrix(water)
    dist_after = geom.distance_matrix(rotated)
    assert np.allclose(dist_before, dist_after)


def test__rotate_in_place(water: Geometry) -> None:
    """Test in-place rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    expected = rot.apply(water.coordinates)
    result = geom.transform.rotate(water, rot, in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, expected)


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


def test__from_xyz_block_empty_raises() -> None:
    """Test that an empty xyz block is rejected."""
    with pytest.raises(XYZFormatError):
        geom.from_xyz_block("", charge=0, spin=0)


def test__xyz_file_roundtrip(water: Geometry, tmp_path: Path) -> None:
    """Test Geometry to xyz file roundtrip."""
    path = tmp_path / "water.xyz"
    water.xyz_file(path=path)
    geo_rt = Geometry.from_xyz_file(path, charge=0, spin=0)

    assert water.symbols == geo_rt.symbols
    assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__xyz_file_function_roundtrip(water: Geometry, tmp_path: Path) -> None:
    """Test module-level xyz_file/from_xyz_file roundtrip."""
    path = tmp_path / "water.xyz"
    geom.xyz_file(water, path=path)
    geo_rt = geom.from_xyz_file(path, charge=0, spin=0)

    assert water.symbols == geo_rt.symbols
    assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__from_xyz_block_classmethod(water: Geometry) -> None:
    """Test Geometry.from_xyz_block classmethod."""
    geo_rt = Geometry.from_xyz_block(water.xyz_block(), charge=0, spin=0)
    assert water.symbols == geo_rt.symbols
    assert np.allclose(water.coordinates, geo_rt.coordinates)


def test__hill_formula_with_carbon() -> None:
    """Test Hill formula with carbon present."""
    methane = Geometry(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        charge=0,
        spin=0,
    )
    assert geom.hill_formula(methane) == "CH4"


def test__hill_formula_no_hydrogen() -> None:
    """Test Hill formula with no hydrogen present."""
    dichlorine = Geometry(
        symbols=["Cl", "Cl"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    assert geom.hill_formula(dichlorine) == "Cl2"


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


def test__view() -> None:
    """Test py3Dmol view construction."""
    water = Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )
    result = geom.view(water, label=True)
    assert result is not None


def test__view_without_label() -> None:
    """Test py3Dmol view construction without atom labels."""
    water = Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )
    result = geom.view(water, label=False)
    assert result is not None


def test__render_svg(water: Geometry, tmp_path: Path) -> None:
    """Test svg rendering."""
    out = tmp_path / "water"
    result = geom.render_svg(water, out=out)
    assert (tmp_path / "water.svg").exists()
    assert "<svg" in str(result)


def test__render_gif(water: Geometry, tmp_path: Path) -> None:
    """Test gif rendering."""
    out = tmp_path / "water"
    geom.render_gif(water, out=out)
    assert (tmp_path / "water.gif").exists()
