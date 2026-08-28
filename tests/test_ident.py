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


def test__hill_formula(water: Geometry) -> None:
    """Test Geometry to Hill-ordered formula."""
    ident = Identity.from_geometry(water, algorithm=Algorithm.HILL_FORMULA)
    assert ident.kind == "formula"
    assert ident.value == "H2O"


def test__hill_formula_with_carbon() -> None:
    """Test Hill formula with carbon present."""
    methane = Geometry(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        charge=0,
        spin=0,
    )
    ident = Identity.from_geometry(methane, algorithm=Algorithm.HILL_FORMULA)
    assert ident.value == "CH4"


def test__hill_formula_no_hydrogen() -> None:
    """Test Hill formula with no hydrogen present."""
    dichlorine = Geometry(
        symbols=["Cl", "Cl"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    ident = Identity.from_geometry(dichlorine, algorithm=Algorithm.HILL_FORMULA)
    assert ident.value == "Cl2"
