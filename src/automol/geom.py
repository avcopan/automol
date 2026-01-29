"""Molecular geometries."""

import hashlib

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import element, rd
from .types import CoordinatesField, FloatArray


class Geometry(BaseModel):
    """
    Molecular geometry.

    Parameters
    ----------
    symbols
        Atomic symbols in order (e.g., ``["H", "O", "H"]``).
        The length of ``symbols`` must match the number of atoms.
    coordinates
        Cartesian coordinates of the atoms in Angstroms.
        Shape is ``(len(symbols), 3)`` and the ordering corresponds to ``symbols``.
    charge
        Total molecular charge.
    spin
        Number of unpaired electrons, i.e. two times the spin quantum number (``2S``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int = 0
    spin: int = 0

    @property
    def masses(self) -> list[float]:
        """Get isotopic masses."""
        return list(map(element.mass, self.symbols))

    @property
    def atomic_numbers(self) -> list[float]:
        """Get atomic numbers."""
        return list(map(element.number, self.symbols))


def from_rdkit_molecule(mol: rd.Mol) -> Geometry:
    """
    Generate geometry from RDKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
        Geometry.
    """
    if not rd.mol.has_coordinates(mol):
        mol = rd.mol.add_coordinates(mol)

    return Geometry(
        symbols=rd.mol.symbols(mol),
        coordinates=rd.mol.coordinates(mol),
        charge=rd.mol.charge(mol),
        spin=rd.mol.spin(mol),
    )


# Properties
def hash(geo: Geometry, decimals: int = 6) -> str:  # noqa: A001
    """
    Generate geometry hash string.

    Parameters
    ----------
    decimals
        Number of decimal places to round the coordinates before hashing.

    Returns
    -------
        Geometry hash string.
    """
    # 1. Convert symbols and coordinates to integers
    numbers = geo.atomic_numbers
    icoords = np.rint(geo.coordinates * 10**decimals)
    # 2. Generate bytes representation of each field
    numbers_bytes = np.asarray(numbers, dtype=np.dtype("<i8")).tobytes("C")
    icoords_bytes = icoords.astype(np.dtype("<i8")).tobytes("C")
    charge_bytes = geo.charge.to_bytes(1, byteorder="little", signed=True)
    spin_bytes = geo.spin.to_bytes(1, byteorder="little", signed=True)
    # 3. Combine all bytes and generate hash
    geo_bytes = b"|".join([numbers_bytes, icoords_bytes, charge_bytes, spin_bytes])
    return hashlib.sha256(geo_bytes).hexdigest()


def center_of_mass(geo: Geometry) -> FloatArray:
    """
    Calculate geometry center of mass.

    Parameters
    ----------
        Geometry.

    Returns
    -------
        Center of mass coordinates.
    """
    masses = list(map(element.mass, geo.symbols))
    coords = geo.coordinates
    return np.sum(np.reshape(masses, (-1, 1)) * coords, axis=0) / np.sum(masses)
