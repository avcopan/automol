"""rdkit mol tests."""

import pytest
from rdkit.Chem import Mol

from automol.rd import mol
from automol.utils.exc import GeometryConversionError


@pytest.fixture
def water() -> Mol:
    """Water fixture."""
    return mol.from_smiles("O", with_coords=True)


def test__smiles_rt() -> None:
    """Test SMILES roundtrip."""
    smiles = "[H]O[H]"
    water = mol.from_smiles(smiles, with_coords=True)

    assert mol.smiles(water) == smiles


def test__inchi_rt() -> None:
    """Test InChI roundtrip."""
    smiles = "InChI=1S/H2O/h1H2"
    water = mol.from_inchi(smiles, with_coords=True)

    assert mol.inchi(water) == smiles


def test__xyz_rt() -> None:
    """Test XYZ roundtrip."""
    xyz = "1\n\nO      0.000000    0.000000    0.000000\n"
    water = mol.from_xyz_block(xyz)

    assert mol.xyz_block(water) == xyz


def test__add_atom_numbers(water: Mol) -> None:
    """Test adding atom numbers."""
    mapping = {0: 1, 1: 2, 2: 0}
    mol.add_atom_numbers(water, to_number=mapping, in_place=True)

    for a in water.GetAtoms():
        assert a.GetProp("atomLabel")


def test__coordinates_without_coordinates_raises() -> None:
    """Test that reading coordinates from a Mol without them is rejected."""
    no_coords = mol.from_smiles("O", with_coords=False)
    with pytest.raises(GeometryConversionError, match="no coordinates"):
        mol.coordinates(no_coords)
