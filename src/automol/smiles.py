"""SMILES strings."""

from . import geom, rd
from .geom import Geometry


def geometry(smi: str) -> Geometry:
    """Get the geometry corresponding to a SMILES string.

    Parameters
    ----------
    smi : str
        Input SMILES string.

    Returns
    -------
        Corresponding geometry.
    """
    mol = rd.mol.from_smiles(smi, with_coords=True)
    return geom.from_rdkit_molecule(mol)
