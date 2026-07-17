"""Identity tests."""

import pytest

from automol import Algorithm, Identity
from automol.ident import AlgorithmRegistry
from automol.utils.exc import AlgorithmAlreadyRegisteredError, UnknownAlgorithmError


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


def test__duplicate_registration_raises() -> None:
    """Test that re-registering an algorithm is rejected."""
    existing = AlgorithmRegistry.get(Algorithm.RDKIT_INCHI)
    with pytest.raises(AlgorithmAlreadyRegisteredError):
        AlgorithmRegistry.register_def(existing)


def test__unknown_algorithm_raises() -> None:
    """Test that looking up an unregistered algorithm is rejected."""
    with pytest.raises(UnknownAlgorithmError):
        AlgorithmRegistry.get("not-a-real-algorithm")  # ty: ignore[invalid-argument-type]


def test__irmsd_algorithm_kind() -> None:
    """Test that IRMSD is tagged with the conformer kind and needs no registry entry."""
    assert Algorithm.IRMSD.kind == "conformer"

    ident = Identity.from_value("3", algorithm=Algorithm.IRMSD)
    assert ident.kind == "conformer"
    assert ident.value == "3"

    with pytest.raises(UnknownAlgorithmError):
        AlgorithmRegistry.get(Algorithm.IRMSD)
