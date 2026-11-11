"""Molecular geometries, identities, and RDKit interoperability.

Deprecated: use autogeom (https://github.com/avcopan/autogeom) instead.
"""

__version__ = "0.0.31"

import warnings

from . import geom, rd
from .geom import Geometry, View
from .ident import (
    Algorithm,
    AlgorithmRegistry,
    IdentityKind,
    hill_formula,
    rdkit_inchi,
    rdkit_smiles,
)

__all__ = [
    "Algorithm",
    "AlgorithmRegistry",
    "Geometry",
    "IdentityKind",
    "View",
    "geom",
    "hill_formula",
    "rd",
    "rdkit_inchi",
    "rdkit_smiles",
]

warnings.warn(
    "automol is deprecated and no longer maintained. Use autogeom instead "
    "(https://github.com/avcopan/autogeom). The identity framework moved to "
    "autostorage (https://github.com/avcopan/autostorage).",
    FutureWarning,
    stacklevel=2,
)
