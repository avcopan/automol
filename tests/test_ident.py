"""Identity tests."""

import pytest

from automol import Algorithm, Geometry, Identity
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


def test__smg_hash_identity_fn(water: Geometry) -> None:
    """Test that SMG_HASH does not raise."""
    ident = Identity.from_geometry(water, algorithm=Algorithm.SMG_HASH)

    assert ident.kind == "conformer"
    assert ident.value.isdigit()
    assert (
        ident.value == Identity.from_geometry(water, algorithm=Algorithm.SMG_HASH).value
    )


def test__smg_hash_invariant(water: Geometry) -> None:
    """Test that SMG_HASH is order invariant."""
    water2 = water.model_copy(deep=True)
    water2.symbols = [water.symbols[i] for i in [2, 0, 1]]
    water2.coordinates = water.coordinates[[2, 0, 1]]

    assert water2.symbols != water.symbols

    hash1 = Identity.from_geometry(water, algorithm=Algorithm.SMG_HASH).value
    hash2 = Identity.from_geometry(water2, algorithm=Algorithm.SMG_HASH).value

    assert hash2 == hash1


def test__smg_hash_distinguishes_geometries(
    water: Geometry, peroxide: Geometry
) -> None:
    """Test that different geometries produce different SMG_HASH identities."""
    water_ident = Identity.from_geometry(water, algorithm=Algorithm.SMG_HASH)
    peroxide_ident = Identity.from_geometry(peroxide, algorithm=Algorithm.SMG_HASH)

    assert water_ident.value != peroxide_ident.value


def test__smg_hash_geometry_not_implemented() -> None:
    """Test that SMG_HASH has no defined inverse back to a Geometry."""
    ident = Identity.from_value("123", algorithm=Algorithm.SMG_HASH)
    with pytest.raises(NotImplementedError):
        ident.geometry()
