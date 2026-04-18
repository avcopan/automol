"""RDKit molecule."""

from collections.abc import Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdDetermineBonds
from rdkit.Chem.rdDistGeom import EmbedMolecule

from ..types import FloatArray


# Importers / Exporters
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


def from_inchi(inchi: str, *, with_coords: bool = False) -> Mol:
    """
    Get RDKit molecule from InChI string.

    Parameters
    ----------
    inchi
        InChI string.

    with_coords, optional
        If `True`, generate 3D coordinates for the molecule.
        If `False` (default), return a molecule without coordinates.

    Returns
    -------
        RDKit molecule.
    """
    mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
    mol = Chem.AddHs(mol)
    if with_coords:
        add_coordinates(mol, in_place=True)
    return mol


def from_xyz_block(xyz_block: str) -> Mol:
    """
    Get RDKit molecule from Geometry.

    Parameters
    ----------
    xyz_block
        Formatted xyz string.

    Returns
    -------
        RDKit molecule.
    """
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    return conn_mol


def smiles(mol: Mol) -> str:
    """
    Get standard SMILES string from Mol.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        SMILES
    """
    return Chem.MolToSmiles(mol)


def inchi(mol: Mol) -> str:
    """
    Get standard InChI string from Mol.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        InChI identifier.
    """
    return Chem.inchi.MolBlockToInchi(Chem.rdmolfiles.MolToMolBlock(mol))


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


def add_atom_numbers(
    mol: Mol, to_number: Mapping[int, int], *, in_place: bool = False
) -> Mol:
    """Add atom numbers.

    Parameters
    ----------
    mol
        RDKit molecule object.
    to_number
        Mapping from atom index to atom number.
    in_place, optional
        If `True`, modify the molecule in place.
        If `False` (default), return a new molecule.

    Returns
    -------
        RDKit molecule object with "atomLabel" property set to f"{symbol}{number}"
    """
    mol = mol if in_place else Mol(mol)
    for atom in mol.GetAtoms():
        number = to_number[atom.GetIdx()]
        symbol = atom.GetSymbol()
        atom.SetProp("atomLabel", f"{symbol}{number}")

    return mol
