"""Identity tests."""

import numpy as np
import pytest
from pydantic import ValidationError

from automol import (
    Algorithm,
    AlgorithmRegistry,
    Geometry,
    IdentityKind,
    hill_formula,
    rdkit_inchi,
    rdkit_smiles,
)
from automol.utils.exc import AlgorithmAlreadyRegisteredError


@pytest.fixture
def water_inchi() -> str:
    """Water identity fixture."""
    return "InChI=1S/H2O/h1H2"


@pytest.fixture
def water_smiles() -> str:
    """Water smiles fixture."""
    return "O"


def test__inchi_roundtrip(water_inchi: str) -> None:
    """Test inchi to Geometry roundtrip."""
    water = rdkit_inchi.geometry_fn(water_inchi)
    water_inchi_rt = rdkit_inchi.identity_fn(water)

    assert water_inchi == water_inchi_rt


def test__smiles_roundtrip(water_smiles: str) -> None:
    """Test smiles to Geometry roundtrip."""
    water = rdkit_smiles.geometry_fn(water_smiles)
    water_smiles_rt = rdkit_smiles.identity_fn(water)

    assert water_smiles == water_smiles_rt


def test__duplicate_registration_raises() -> None:
    """Test that re-registering an algorithm is rejected."""
    with pytest.raises(AlgorithmAlreadyRegisteredError):
        AlgorithmRegistry.register(
            name="rdkit inchi",
            kind=IdentityKind.STEREOISOMER,
            identity_fn=rdkit_inchi.identity_fn,
        )


def test__hill_formula(water: Geometry) -> None:
    """Test Geometry to Hill-ordered formula."""
    ident = hill_formula.identity_fn(water)
    assert ident == "H2O"


def test__hill_formula_with_carbon() -> None:
    """Test Hill formula with carbon present."""
    methane = Geometry(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        charge=0,
        spin=0,
    )
    ident = hill_formula.identity_fn(methane)
    assert ident == "CH4"


def test__hill_formula_no_hydrogen() -> None:
    """Test Hill formula with no hydrogen present."""
    dichlorine = Geometry(
        symbols=["Cl", "Cl"],
        coordinates=[[0, 0, 0], [2, 0, 0]],
        charge=0,
        spin=0,
    )
    ident = hill_formula.identity_fn(dichlorine)
    assert ident == "Cl2"


def test__smiles_parent_algorithm() -> None:
    """Test that RDKitSMILES uses RDKitInChI as its parent algorithm."""
    canon_smiles = "CCCCC"
    weird_smiles = "C(C)CCC"
    smiles = [weird_smiles, "CCC", "CC(C)C"]

    geo = rdkit_smiles.geometry_fn(canon_smiles)
    other_geos = {s: rdkit_smiles.geometry_fn(s) for s in smiles}

    canon_ident = rdkit_smiles.identity_fn(geo)
    weird_ident = rdkit_smiles.identity_fn(geo, other_geos=other_geos)

    assert canon_ident == canon_smiles
    assert weird_ident == weird_smiles


def test__smiles_without_matching_other_geo(water: Geometry) -> None:
    """Test that canonical SMILES is returned when no `other_geos` entry matches."""
    methane = rdkit_smiles.geometry_fn("C")
    assert rdkit_smiles.identity_fn(water, other_geos={"C": methane}) == "O"


def test__hill_formula_without_carbon_is_alphabetical() -> None:
    """Test that, without carbon, H is ordered alphabetically (strict Hill order)."""
    hcl = Geometry(symbols=["H", "Cl"], coordinates=np.eye(2, 3), charge=0, spin=0)
    assert hill_formula.identity_fn(hcl) == "ClH"

    nh3 = Geometry(
        symbols=["N", "H", "H", "H"], coordinates=np.eye(4, 3), charge=0, spin=0
    )
    assert hill_formula.identity_fn(nh3) == "H3N"


def test__hill_formula_with_carbon_and_heteroatoms() -> None:
    """Test that, with carbon, C and H come first, then the rest alphabetically."""
    symbols = ["Cl", "C", "H", "H", "H", "Br", "N"]
    geo = Geometry(symbols=symbols, coordinates=np.eye(7, 3), charge=0, spin=0)
    assert hill_formula.identity_fn(geo) == "CH3BrClN"


@pytest.mark.parametrize(
    ("algorithm", "value"),
    [(rdkit_smiles, "C1CC"), (rdkit_inchi, "not an inchi")],
)
def test__geometry_fn_invalid_value_raises(algorithm: Algorithm, value: str) -> None:
    """Test that invalid identifiers raise a clear error."""
    with pytest.raises(ValueError, match="Invalid"):
        algorithm.geometry_fn(value)


def test__geometry_fn_reproducible(water_smiles: str) -> None:
    """Test that identifier -> Geometry is reproducible."""
    assert rdkit_smiles.geometry_fn(water_smiles) == rdkit_smiles.geometry_fn(
        water_smiles
    )


def test__algorithm_rejects_incompatible_signatures() -> None:
    """Test that identity/geometry functions with wrong signatures are rejected."""

    def bad_identity_fn(geo: Geometry) -> str:
        return str(geo)

    def bad_geometry_fn(value: str, extra: int) -> Geometry:
        raise NotImplementedError(value, extra)

    with pytest.raises(ValidationError, match="incompatible signature"):
        Algorithm(
            name="bad identity",
            kind=IdentityKind.FORMULA,
            identity_fn=bad_identity_fn,
        )
    with pytest.raises(ValidationError, match="incompatible signature"):
        Algorithm(
            name="bad geometry",
            kind=IdentityKind.FORMULA,
            identity_fn=hill_formula.identity_fn,
            geometry_fn=bad_geometry_fn,
        )
