"""Identity tests."""

import pytest

from automol import Identity


@pytest.fixture
def water_inchi() -> Identity:
    """Water identity fixture."""
    return Identity(
        kind="stereoisomer", algorithm="rdkit inchi", value="InChI=1S/H2O/h1H2"
    )


@pytest.fixture
def water_smiles() -> Identity:
    """Water smiles fixture."""
    return Identity(kind="stereoisomer", algorithm="rdkit smiles", value="O")


def test__inchi_roundtrip(water_inchi: Identity) -> None:
    """Test inchi to Geometry roundtrip."""
    water = water_inchi.geometry()
    water_inchi_rt = Identity.from_geometry(water, algorithm="rdkit inchi")

    assert water_inchi.kind == water_inchi_rt.kind
    assert water_inchi.value == water_inchi_rt.value


def test__smiles_roundtrip(water_smiles: Identity) -> None:
    """Test smiles to Geometry roundtrip."""
    water = water_smiles.geometry()
    water_smiles_rt = Identity.from_geometry(water, algorithm="rdkit smiles")

    assert water_smiles.kind == water_smiles_rt.kind
    assert water_smiles.value == water_smiles_rt.value
