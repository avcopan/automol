"""Molecular geometries."""

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import element, rd
from .types import CoordinatesField, FloatArray


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
