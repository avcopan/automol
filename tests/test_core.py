"""Geometry core model tests."""

from pathlib import Path

import numpy as np
import pytest

from automol import Geometry, geom, rd
from automol.utils.exc import XYZFormatError


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


def test__xyz_roundtrip(water: Geometry) -> None:
    """Test Geometry to xyz string roundtrip."""
    xyz = water.xyz_block()
    geo_rt = geom.from_xyz_block(xyz, charge=0, spin=0)

    assert water.symbols == geo_rt.symbols
    assert np.allclose(water.coordinates, geo_rt.coordinates)


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


def test__from_xyz_block_malformed_line_raises() -> None:
    """Test that a malformed xyz line is rejected."""
    bad_block = "1\n\nnot a valid xyz line\n"
    with pytest.raises(XYZFormatError):
        geom.from_xyz_block(bad_block, charge=0, spin=0)


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
