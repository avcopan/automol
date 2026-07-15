"""Identity tests."""

import pytest

from automol import Algorithm, Identity


@pytest.fixture
def water_inchi() -> Identity:
    """Water identity fixture."""
    return Identity.from_value("InChI=1S/H2O/h1H2", algorithm=Algorithm.RDKIT_INCHI)


@pytest.fixture
def water_smiles() -> Identity:
    """Water smiles fixture."""
    return Identity.from_value("O", algorithm=Algorithm.RDKIT_SMILES)


def test__inchi_roundtrip(water_inchi: Identity) -> None:
    """Test inchi to Geometry roundtrip."""
    water = water_inchi.geometry()
    water_inchi_rt = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_INCHI)

    assert water_inchi.kind == water_inchi_rt.kind
    assert water_inchi.value == water_inchi_rt.value


def test__smiles_roundtrip(water_smiles: Identity) -> None:
    """Test smiles to Geometry roundtrip."""
    water = water_smiles.geometry()
    water_smiles_rt = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_SMILES)

    assert water_smiles.kind == water_smiles_rt.kind
    assert water_smiles.value == water_smiles_rt.value


def test__kind_mismatch_raises() -> None:
    """Test that an explicit mismatched kind is rejected."""
    with pytest.raises(ValueError, match="belongs to kind"):
        Identity(algorithm=Algorithm.RDKIT_INCHI, value="x", kind="conformer")
