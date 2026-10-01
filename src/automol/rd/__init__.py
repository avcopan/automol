"""RDKit bridge, operating on RDKit `Mol` objects only (never `Geometry`)."""

from . import mol
from .mol import Mol

__all__ = ["Mol", "mol"]
