"""Graph tests."""

import pytest

from automol import graph


def test__smiles() -> None:
    """Test graph smiles."""
    water_smiles = "O"
    water_inchi = "InChI=1S/H2O/h1H2"
    water = graph.from_smiles(water_smiles)
    assert graph.inchi(water) == water_inchi


def test__inchi() -> None:
    """Test graph inchi."""
    water_inchi = "InChI=1S/H2O/h1H2"
    water = graph.from_inchi(water_inchi)
    assert graph.inchi(water) == water_inchi


def test__remove_bonds() -> None:
    """Test graph remove bonds."""
    water_smiles = "O"
    oh_h_smiles = "[OH].[H]"
    water = graph.from_smiles(water_smiles)
    oh_h_ref = graph.from_smiles(oh_h_smiles)
    oh_h = graph.remove_bonds(water, [(0, 1)])
    assert graph.is_isomorphic(oh_h, oh_h_ref)


def test__symbols() -> None:
    """Test graph symbols."""
    water_smiles = "O"
    water = graph.from_smiles(water_smiles)
    assert graph.symbols(water) == ["O", "H", "H"]


@pytest.mark.parametrize(
    ("smi", "ref"),
    [
        ("[H]", [1]),
        ("[He]", [0]),
        ("O", [0, 0, 0]),
        ("[OH]", [1, 0]),
        ("C=C", [1, 1, 0, 0, 0, 0]),
        ("C#C", [2, 2, 0, 0]),
        ("O[O]", [0, 1, 0]),
    ],
)
def test__open_valences(smi: str, ref: list[int]) -> None:
    """Test graph open valences."""
    gra = graph.from_smiles(smi)
    assert graph.open_valences(gra) == ref
