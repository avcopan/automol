"""Core element interface."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Element:
    """
    Chemical element.

    Attributes
    ----------
    Z :
        Atomic number.
    A :
        Mass number.
    symbol :
        Chemical symbol.
    mass :
        Atomic mass.
    """

    Z: int
    A: int
    group: int
    period: int
    symbol: str
    mass: float
    covalent_radius: float
    nvalence: int


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
    """
    Retrieve element by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Requested element.

    Raises
    ------
    TypeError
        If key is not int or str.
    """
    if isinstance(key, int):
        return ELEMENT_BY_NUMBER[key]

    if isinstance(key, str):
        return ELEMENT_BY_SYMBOL[key.casefold()]

    msg = f"Element key must be int or str, got {type(key).__name__}"
    raise TypeError(msg)


def number(key: int | str) -> int:
    """
    Retrieve atomic number of element by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Atomic number.
    """
    return from_key(key).Z


def mass_number(key: int | str) -> int:
    """
    Retrieve mass number of element by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Mass number.
    """
    return from_key(key).A


def symbol(key: int | str) -> str:
    """
    Retrieve atomic symbol of element by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Atomic symbol.
    """
    return from_key(key).symbol


def mass(key: int | str) -> float:
    """
    Retrieve atomic mass of element by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Atomic mass.
    """
    return from_key(key).mass


def covalent_radius(key: int | str) -> float:
    """
    Retrieve covalent radius of element by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Covalent radius.
    """
    return from_key(key).covalent_radius


def group(key: int | str) -> int:
    """
    Retrieve group number by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Group number.
    """
    return from_key(key).group


def period(key: int | str) -> int:
    """
    Retrieve period number by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Period.
    """
    return from_key(key).period


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
    """
    Determine shell capacity by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).

    Returns
    -------
        Shell capacity.
    """
    return PERIOD_SHELL_CAPACITY[period(key)]


def nvalence(key: int | str, *, override: dict[str, int] | None = None) -> int:
    """
    Retrieve number of valence electrons by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).
    override :
        Dictionary of valence overrides by atomic symbol.

    Returns
    -------
        Number of valence electrons.
    """
    if override is not None:
        symb = from_key(key).symbol
        if symb in override:
            return override[symb]
    return from_key(key).nvalence


def bonding_capacity(key: int | str, *, override: dict[str, int] | None = None) -> int:
    """
    Determine bonding capacity by atomic number or symbol.

    Parameters
    ----------
    key :
        Atomic number (int) or symbol (str).
    override :
        Dictionary of bonding capacity overrides by atomic symbol.

    Returns
    -------
        Bonding capacity.
    """
    if override is not None:
        symb = from_key(key).symbol
        if symb in override:
            return override[symb]
    cap = shell_capacity(key)
    nval = nvalence(key)
    return min(nval, cap - nval)
