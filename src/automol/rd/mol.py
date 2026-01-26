"""RDKit molecule."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule

from ..types import FloatArray


# Generators
def from_smiles(smi: str, *, with_coords: bool = False) -> Mol:
    """
    Get RDKit molecule from SMILES string.

    Parameters
    ----------
    smi
        SMILES string.

    with_coords, optional
        If `True`, generate 3D coordinates for the molecule.
        If `False` (default), return a molecule without coordinates.

    Returns
    -------
        RDKit molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if with_coords:
        add_coordinates(mol, in_place=True)
    return mol


# Properties
def symbols(mol: Mol) -> list[str]:
    """
    Get atomic symbols.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        List of atomic symbols.
    """
    return [a.GetSymbol() for a in mol.GetAtoms()]


def coordinates(mol: Mol) -> FloatArray:
    """
    Get atom coordinates.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        Atomic coordinates as (N, 3) numpy array.

    Raises
    ------
    ValueError
        If the molecule has no coordinates.
    """
    if not has_coordinates(mol):
        msg = "Molecule has no coordinates. Did you forget to add them?"
        raise ValueError(msg)

    natms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(natms)]
    return np.array(coords, dtype=np.float64)


def charge(mol: Mol) -> int:
    """
    Get molecular charge.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        Molecular charge as an integer.
    """
    return Chem.GetFormalCharge(mol)


def spin(mol: Mol) -> int:
    """
    Get molecular spin (number of unpaired electrons).

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        Number of unpaired electrons as an integer.
    """
    return Descriptors.NumRadicalElectrons(mol)


# Boolean properties
def has_coordinates(mol: Mol) -> bool:
    """
    Check if coordinates have been added.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        `True` if the molecule has coordinates, False otherwise.
    """
    return bool(mol.GetNumConformers())


# Transformations
def add_coordinates(mol: Mol, *, in_place: bool = False) -> Mol:
    """
    Add coordinates, if missing.

    Parameters
    ----------
    mol
        RDKit molecule object.
    in_place, optional
        If `True`, modify the molecule in place.
        If `False` (default), return a new molecule.

    Returns
    -------
        RDKit molecule object with coordinates.
        (Unmodified if it already had coordinates.)
    """
    if has_coordinates(mol):
        return mol

    mol = mol if in_place else Mol(mol)
    EmbedMolecule(mol)
    return mol
