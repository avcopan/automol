"""Core element interface."""

import json
from dataclasses import dataclass
from pathlib import Path


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
    symbol: str
    mass: float


ELEMENT_BY_NUMBER: dict[int, Element] = {}
ELEMENT_BY_SYMBOL: dict[str, Element] = {}


def _load_elements() -> None:
    data_path = Path(__file__).with_name("elements-data.json")

    with data_path.open() as f:
        elements_data: list[dict[str, object]] = json.load(f)

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
