"""Geometry tests."""

import pytest

from automol import Geometry, Identity


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # ty:ignore[invalid-argument-type]
    )


@pytest.mark.parametrize(
    argnames=("algorithm", "value", "spin"),
    argvalues=[
        ("rdkit smiles", "F/C=C/F", 0),
        ("rdkit smiles", "[OH]", 1),
        ("rdkit inchi", "InChI=1S/C2H2F2/c3-1-2-4/h1-2H/b2-1+", 0),
        ("rdkit inchi", "InChI=1S/HO/h1H", 1),
    ],
)
def test__stereoisomer_roundtrip(algorithm: str, value: str, spin: str) -> None:
    """Test value -> Identity -> Geometry -> value roundtrip."""
    identity = Identity(kind="stereoisomer", algorithm=algorithm, value=value)

    geo = identity.geometry()
    assert geo.spin == spin

    identity2 = Identity.from_geometry(geo, algorithm=algorithm)
    assert identity == identity2


def test__from_geometry(water: Geometry) -> None:
    """Test identity generation from Geometry."""
    smiles = Identity.from_geometry(water, algorithm="rdkit smiles")
    assert smiles.value == "O"

    inchi = Identity.from_geometry(water, algorithm="rdkit inchi")
    assert inchi.value == "InChI=1S/H2O/h1H2"
