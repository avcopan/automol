"""Molecular geometry functions."""

from typing import TYPE_CHECKING

import numpy as np
from scipy import spatial
from scipy.sparse.csgraph import connected_components

from ..utils import element
from ..utils.types import FloatArray

if TYPE_CHECKING:
    from .core import Geometry


def center_of_mass(geo: "Geometry") -> FloatArray:
    """Calculate the geometry center of mass.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
    FloatArray
        Center of mass coordinates.
    """
    masses = list(map(element.mass, geo.symbols))
    coords = geo.coordinates
    return np.sum(np.reshape(masses, (-1, 1)) * coords, axis=0) / np.sum(masses)


def distance_matrix(geo: "Geometry") -> np.ndarray:
    """Calculate the geometry distance matrix.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
    np.ndarray
        Distance matrix of geometry.
    """
    return spatial.distance.cdist(geo.coordinates, geo.coordinates)


def adjacency_matrix(
    geo: "Geometry", *, sigma: float = 1.3, flood_fill: bool = False
) -> np.ndarray:
    """Compute the molecular adjacency matrix based on covalent radii.

    An edge exists between two atoms if their distance is less than `sigma`
    times the sum of their covalent radii.

    Parameters
    ----------
    geo
        Geometry.
    sigma
        Scaling factor applied to the sum of covalent radii.
    flood_fill
        If True, increase sigma by 0.05 until A is continuous (one fragment).

    Returns
    -------
    np.ndarray
        2D binary adjacency matrix.
    """
    radii = np.array(geo.covalent_radii)
    dmat = distance_matrix(geo)

    while True:
        umat = sigma * (radii[:, None] + radii[None, :])
        amat = (dmat < umat).astype(int)
        np.fill_diagonal(amat, 0)

        if not flood_fill:
            break

        n_components, _ = connected_components(amat, directed=False)
        if n_components <= 1:
            break
        sigma += 0.05

    return amat


def distance_keys(geo: "Geometry") -> np.ndarray:
    """Generate a sorted distance descriptor.

    The descriptor characterizes a geometry using the distance between all unique
    atom pairs, sorted first by atomic numbers (z_low, z_high) then their pairwise
    distances.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
    np.ndarray
        Array where each row contains [z1, z2, distance]
    """
    z = np.asarray(geo.atomic_numbers)
    dmat = distance_matrix(geo)
    iu, ju = np.triu_indices(geo.atom_count, k=1)

    pairs = np.sort(np.stack([z[iu], z[ju]], axis=1), axis=1)
    key = np.column_stack([pairs.astype(float), dmat[iu, ju]])

    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    return key[order]
