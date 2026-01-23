"""Model and functions for manipulating molecular geometries."""

from pydantic import BaseModel, ConfigDict

from .types import CoordinatesField


class Geometry(BaseModel):
    """
    Molecular geometry.

    Parameters
    ----------
    symbols : list[str]
        Atomic symbols in order (e.g., ``["H", "O", "H"]``).
        The length of ``symbols`` must match the number of atoms.

    coordinates : CoordinatesField
        Cartesian coordinates of the atoms.
        Shape is ``(len(symbols), 3)`` and the ordering corresponds to ``symbols``.
        Units are assumed to be **angstroms** unless otherwise specified.

    charge : int, optional
        Total molecular charge.
        Default is ``0``.

    spin : int, optional
        Total spin multiplicity or spin quantum number (depending on convention).
        Default is ``0``.
    """

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int = 0
    spin: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)
