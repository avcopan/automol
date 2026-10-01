"""Core element interface."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exc import ElementNotFoundError


@dataclass(frozen=True, slots=True)
class Element:
    """Chemical element.

    Attributes:
        Z (int): Atomic number.
        A (int): Mass number.
        group (int | None): Group number, or `None` for lanthanides and
            actinides.
        period (int): Period number.
        symbol (str): Chemical symbol.
        mass (float): Atomic mass.
        covalent_radius (float): Pyykko covalent radius, in Angstroms.
        valence (int): Number of valence electrons.
        electronegativity (float | None): Pauling electronegativity, if defined.
        metal (bool): Whether the element is a metal.
    """

    Z: int
    A: int
    group: int | None
    period: int
    symbol: str
    mass: float
    covalent_radius: float
    valence: int
    electronegativity: float | None
    metal: bool


ELEMENT_BY_NUMBER: dict[int, Element] = {}
ELEMENT_BY_SYMBOL: dict[str, Element] = {}


def _load_elements() -> None:
    data_path = Path(__file__).with_name("elements-data.json")

    with data_path.open() as f:
        elements_data: list[dict[str, Any]] = json.load(f)

    for element_data in elements_data:
        element = Element(**element_data)

        ELEMENT_BY_NUMBER[element.Z] = element
        ELEMENT_BY_SYMBOL[element.symbol.casefold()] = element


_load_elements()


def from_key(key: int | str) -> Element:
    """Retrieve element by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        Element: Requested element.

    Raises:
        TypeError: If `key` is not an `int` or `str`.
        ElementNotFoundError: If no element matches the given atomic number or
            symbol.
    """
    if isinstance(key, int):
        if key not in ELEMENT_BY_NUMBER:
            msg = f"No element with atomic number {key!r}."
            raise ElementNotFoundError(msg)
        return ELEMENT_BY_NUMBER[key]

    if isinstance(key, str):
        symb = key.casefold()
        if symb not in ELEMENT_BY_SYMBOL:
            msg = f"No element with symbol {key!r}."
            raise ElementNotFoundError(msg)
        return ELEMENT_BY_SYMBOL[symb]

    msg = f"Element key must be int or str, got {type(key).__name__}"
    raise TypeError(msg)


def number(key: int | str) -> int:
    """Retrieve atomic number of element by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        int: Atomic number.
    """
    return from_key(key).Z


def mass_number(key: int | str) -> int:
    """Retrieve mass number of element by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        int: Mass number.
    """
    return from_key(key).A


def symbol(key: int | str) -> str:
    """Retrieve atomic symbol of element by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        str: Atomic symbol.
    """
    return from_key(key).symbol


def mass(key: int | str) -> float:
    """Retrieve atomic mass of element by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        float: Atomic mass.
    """
    return from_key(key).mass


def covalent_radius(key: int | str) -> float:
    """Retrieve covalent radius of element by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        float: Covalent radius.
    """
    return from_key(key).covalent_radius


def group(key: int | str) -> int | None:
    """Retrieve group number by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        int | None: Group number, or `None` for lanthanides and actinides.
    """
    return from_key(key).group


def period(key: int | str) -> int:
    """Retrieve period number by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        int: Period number.
    """
    return from_key(key).period


def electronegativity(key: int | str) -> float | None:
    """Retrieve Pauling electronegativity by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        float | None: Pauling electronegativity, or `None` if undefined (for
            example, for He, Ne, or Ar).
    """
    return from_key(key).electronegativity


def is_metal(key: int | str) -> bool:
    """Determine whether an element is a metal by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        bool: `True` for alkali, alkaline earth, poor, and transition metals,
        lanthanides, and actinides; `False` otherwise.
    """
    return from_key(key).metal


PERIOD_SHELL_CAPACITY = {
    1: 2,  # H, He
    2: 8,  # Li to Ne
    3: 8,  # Na to Ar
    4: 18,  # K to Kr
    5: 18,  # Rb to Xe
    6: 32,  # Cs to Rn
    7: 32,  # Fr to Og
}


def shell_capacity(key: int | str) -> int:
    """Determine shell capacity by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).

    Returns:
        int: Shell capacity.
    """
    return PERIOD_SHELL_CAPACITY[period(key)]


def valence(key: int | str, *, override: dict[str, int] | None = None) -> int:
    """Retrieve number of valence electrons by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).
        override: Dictionary of valence overrides by atomic symbol.

    Returns:
        int: Number of valence electrons.
    """
    if override is not None:
        symb = from_key(key).symbol
        if symb in override:
            return override[symb]
    return from_key(key).valence


def bonding_capacity(key: int | str, *, override: dict[str, int] | None = None) -> int:
    """Determine bonding capacity by atomic number or symbol.

    Args:
        key: Atomic number (`int`) or symbol (`str`).
        override: Dictionary of bonding capacity overrides by atomic symbol.

    Returns:
        int: Bonding capacity.
    """
    if override is not None:
        symb = from_key(key).symbol
        if symb in override:
            return override[symb]
    cap = shell_capacity(key)
    nval = valence(key)
    return min(nval, cap - nval)
