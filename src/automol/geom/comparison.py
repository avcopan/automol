"""Geometry comparison functions."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import irmsd

if TYPE_CHECKING:
    from .core import Geometry

RMSD_THRESHOLD = 0.125


def is_duplicate_conformer(
    geo: "Geometry",
    geos: Sequence["Geometry"],
    *,
    rthr: float = RMSD_THRESHOLD,
) -> list[bool]:
    """Check whether a geometry is an identical conformer to one in a list.

    Two geometries are considered identical conformers if, after optimal
    alignment (translation, rotation, and atom matching), their interatomic
    RMSD is below `rthr`. Candidates with a different atom count are never
    considered a match.

    Parameters
    ----------
    geo
        Geometry to check.
    geos
        Candidate geometries to compare against.
    rthr
        iRMSD threshold, in Angstroms, below which two geometries are
        considered identical conformers.

    Returns
    -------
        List the same length as `geos`, with `True` at each position where `geo`
        matches the corresponding candidate, `False` otherwise.
    """
    mol = irmsd.Molecule(symbols=geo.symbols, positions=geo.coordinates)

    matches = []
    for candidate in geos:
        if len(candidate.symbols) != len(geo.symbols):
            matches.append(False)
            continue
        candidate_mol = irmsd.Molecule(
            symbols=candidate.symbols, positions=candidate.coordinates
        )
        value, _, _ = irmsd.get_irmsd_molecule(mol, candidate_mol)
        matches.append(value <= rthr)

    return matches
