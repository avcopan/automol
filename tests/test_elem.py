"""Elements tests."""

import pytest

from automol import element


@pytest.mark.parametrize(("key", "mass_number"), [("H", 1), (115, 287)])
def test__mass_number(key: str | int, mass_number: int) -> None:
    """Test retrieval of element mass number."""
    assert element.mass_number(key) == mass_number


@pytest.mark.parametrize(("key", "symbol"), [("H", "H"), (115, "Mc")])
def test__symbol(key: str | int, symbol: str) -> None:
    """Test retrieval of element symbol."""
    assert element.symbol(key) == symbol


@pytest.mark.parametrize(("key", "mass"), [("H", 1.008), (115, 287.191)])
def test__mass(key: str | int, mass: float) -> None:
    """Test retrieval of element mass."""
    assert round(element.mass(key), 3) == mass


@pytest.mark.parametrize(("key", "radius"), [("H", 0.32), (115, 1.62)])
def test__covalent_radius(key: str | int, radius: float) -> None:
    """Test retrieval of element covalent radius."""
    assert element.covalent_radius(key) == radius


@pytest.mark.parametrize(("key", "group"), [("H", 1), (115, 15)])
def test__group(key: str | int, group: int) -> None:
    """Test retrieval of element group."""
    assert element.group(key) == group


@pytest.mark.parametrize(("key", "period"), [("H", 1), (115, 7)])
def test__period(key: str | int, period: int) -> None:
    """Test retrieval of element period."""
    assert element.period(key) == period


@pytest.mark.parametrize(("key", "capacity"), [("H", 2), (115, 32)])
def test__shell_capacity(key: str | int, capacity: int) -> None:
    """Test retrieval of element shell capacity."""
    assert element.shell_capacity(key) == capacity


@pytest.mark.parametrize(
    ("key", "override", "nvalence"), [("H", None, 1), (115, {"Mc": 4}, 4)]
)
def test__nvalence(
    key: str | int, override: dict[str, int] | None, nvalence: int
) -> None:
    """Test retrieval of element nvalence."""
    assert element.valence(key, override=override) == nvalence


@pytest.mark.parametrize(
    ("key", "override", "capacity"), [("H", None, 1), (115, {"Mc": 6}, 6)]
)
def test__bonding_capacity(
    key: str | int, override: dict[str, int] | None, capacity: int
) -> None:
    """Test retrieval of element bonding capacity."""
    assert element.bonding_capacity(key, override=override) == capacity
