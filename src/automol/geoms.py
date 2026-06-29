"""analyze geometries."""

import itertools
from collections.abc import Sequence

import numpy as np

from automol.utils.types import FloatArray

from . import geom
from .geom import Geometry


# Comparison
def kabsch(
    geo1: Geometry, geo2: Geometry, *, heavy_only: bool = False
) -> tuple[FloatArray, FloatArray, float]:
    """
    Compute the optimal rotation / translation to align two Geometries and their RMSD.

    For more information on the numerical method, see https://hunterheidenreich.com/posts/kabsch-algorithm/

    Parameters
    ----------
    geo1
        Geometry object.
    geo2
        Geometry object.
    heavy_only
        If True, only consider heavy atoms.

    Returns
    -------
    FloatArray
        Optimal rotation of geo2 onto geo1
    FloatArray
        Optimal translation of geo2 onto geo1
    float
        RMSD
    """
    p = np.array(geo1.coordinates)
    q = np.array(geo2.coordinates)
    p_masses = geo1.masses
    q_masses = geo2.masses

    if heavy_only:
        mask_p = np.array([s != "H" for s in geo1.symbols])
        mask_q = np.array([s != "H" for s in geo2.symbols])
        # Contrapositive of "If no heavy atoms exist (e.g., H2, H), skip masking"
        if np.any(mask_p):
            p, q = p[mask_p], q[mask_q]

            p_masses = np.asanyarray(p_masses)
            q_masses = np.asanyarray(q_masses)

            p_masses, q_masses = p_masses[mask_p], q_masses[mask_q]

    if p.shape != q.shape:
        msg = f"""
        Input arrays must have same number of dimensions.\n
        {p.shape = }\n
        {q.shape = }\n
        """
        raise ValueError(msg)

    # --- Optimal translation -------------------
    centroid_p = geom.center_of_mass(geo1)
    centroid_q = geom.center_of_mass(geo2)
    t = centroid_p - centroid_q  # Optimal translation
    # Center the coordinates
    p = p - centroid_p
    q = q - centroid_q

    # --- Optimal rotation ----------------------
    H = np.dot(p.T, q)  # Covariance matrix  # noqa: N806
    U, _, Vt = np.linalg.svd(H)  # noqa: N806

    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:  # Validate right-handed coordinates
        Vt[-1, :] *= -1.0

    R = np.dot(Vt.T, U.T)  # Optimal rotation  # noqa: N806

    # --- RMSD ----------------------------------
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / p.shape[0])

    return R, t, rmsd


# Multi-geometry operations
def concat(geos: Sequence[Geometry]) -> Geometry:
    """Concatenate geometries.

    Parameters
    ----------
    geos
        List of geometries.

    Returns
    -------
        Geometry.
    """
    symbols = list(itertools.chain.from_iterable(geo.symbols for geo in geos))
    coordinates = np.vstack([geo.coordinates for geo in geos])
    charge = sum(geo.charge for geo in geos)
    spin = sum(geo.spin for geo in geos)
    return Geometry(symbols=symbols, coordinates=coordinates, charge=charge, spin=spin)
