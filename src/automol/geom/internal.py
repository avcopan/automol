"""Internal-coordinate extraction and editing (bonds, angles, dihedrals)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from .properties import distance_matrix

if TYPE_CHECKING:
    from .core import Geometry


def bonds(
    geo: "Geometry", amat: np.ndarray[tuple[int, int], np.dtype[np.bool]]
) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.number]]:
    """Compile distances of bonded doubles.

    Parameters
    ----------
    geo
        Geometry.
    amat
        Adjacency matrix.

    Returns
    -------
    np.ndarray
        List of bonded doubles and their distances [[z1, z2, dist], ...].

    Example
    -------
    >>> from automol import Geometry
    >>> from automol.geom import adjacency_matrix
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> amat = adjacency_matrix(geo)
    >>> b = bonds(geo, amat)
    >>> b.shape
    (2, 3)
    >>> [tuple(float(x) for x in row) for row in b]
    [(1.0, 8.0, 1.0), (1.0, 8.0, 1.0)]
    """
    zs = geo.atomic_numbers

    i, j = np.where(np.triu(amat) == 1)
    dmat = distance_matrix(geo)

    results = []
    for a1, a2 in zip(i, j, strict=True):
        z1, z2 = zs[a1], zs[a2]
        if z1 > z2:
            z1, z2 = z2, z1
        results.append((z1, z2, float(dmat[a1, a2])))

    return np.array(sorted(results))


def angles(
    geo: "Geometry", amat: np.ndarray[tuple[int, int], np.dtype[np.bool]]
) -> np.ndarray[tuple[int, Literal[4]], np.dtype[np.number]]:
    """Compile angles of bonded triples.

    Parameters
    ----------
    geo
        Geometry.
    amat
        Adjacency matrix.

    Returns
    -------
    array
        List of bonded triples and their angles [[z1, z2, z3, theta], ...].

    Example
    -------
    >>> from automol import Geometry
    >>> from automol.geom import adjacency_matrix
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> amat = adjacency_matrix(geo)
    >>> a = angles(geo, amat)
    >>> a.shape
    (1, 4)
    >>> round(float(a[0][-1]), 6)
    1.570796
    """
    zs = geo.atomic_numbers

    i, j, k = np.where((amat[:, :, None] == 1) & (amat[None, :, :] == 1))
    i, j, k = i[i < k], j[i < k], k[i < k]

    b1 = geo.coordinates[j] - geo.coordinates[i]
    b2 = geo.coordinates[j] - geo.coordinates[k]

    n1: float = np.linalg.norm(b1, axis=1)
    n2: float = np.linalg.norm(b2, axis=1)

    cos_theta = np.clip(np.einsum("nt,nt->n", b1, b2) / (n1 * n2), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    results = []
    for a1, a2, a3, th in zip(i, j, k, theta, strict=True):
        z1, z2, z3 = zs[a1], zs[a2], zs[a3]
        if z1 > z3:
            z1, z3 = z3, z1
        results.append((z1, z2, z3, float(th)))

    return np.array(sorted(results))


def dihedrals(
    geo: "Geometry", amat: np.ndarray[tuple[int, int], np.dtype[np.bool]]
) -> np.ndarray[tuple[int, Literal[5]], np.dtype[np.number]]:
    """Compile dihedrals of bonded quadruples.

    Parameters
    ----------
    geo
        Geometry.
    amat
        Adjacency matrix.

    Returns
    -------
    array
        List of bonded quadruples and their dihedrals [[i, j, k, l, phi], ...].

    Example
    -------
    >>> import numpy as np
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["H", "O", "O", "H"],
    ...     coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> amat = np.array(
    ...     [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    ... )
    >>> d = dihedrals(geo, amat)
    >>> d.shape
    (1, 5)
    >>> round(float(d[0][-1]), 6)
    1.570796
    """
    i, j, k, l = np.where(  # noqa: E741
        (amat[:, :, None, None] == 1)
        & (amat[None, :, :, None] == 1)
        & (amat[None, None, :, :] == 1)
    )

    valid = (i != k) & (j != l) & (i != l) & (i < l)
    i, j, k, l = i[valid], j[valid], k[valid], l[valid]  # noqa: E741

    b1 = geo.coordinates[j] - geo.coordinates[i]
    b2 = geo.coordinates[k] - geo.coordinates[j]
    b3 = geo.coordinates[l] - geo.coordinates[k]

    nb2 = np.linalg.norm(b2, axis=1, keepdims=True)
    nb2 = np.where(nb2 == 0.0, 1.0, nb2)
    ub2 = b2 / nb2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    cross12 = np.cross(n1, n2)

    x = np.einsum("nt,nt->n", n1, n2)
    y = np.einsum("nt,nt->n", cross12, ub2)
    phi = np.arctan2(y, x)

    return np.column_stack([i, j, k, l, phi])


def set_distance(
    geo: "Geometry",
    *,
    idxs: Sequence[int],
    val: float,
    max_change: float = 0.25,
    in_place: bool = False,
) -> "Geometry":
    """
    Set distance between two atoms.

    Parameters
    ----------
    geo
        Geometry object.
    idxs
        Atom indices.
    val
        Value of new distance.
    max_change
        Max allowable change in distance.
    in_place
        Modify the geometry in place.

    Returns
    -------
    Geometry
        Updated geometry.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> updated = set_distance(geo, idxs=[0, 1], val=1.2)
    >>> dist = float(np.linalg.norm(updated.coordinates[1] - updated.coordinates[0]))
    >>> round(dist, 6)
    1.2
    """
    if len(idxs) != 2:  # noqa: PLR2004
        msg = f"Wrong number of indices provided ({len(idxs)} != 2)."
        raise ValueError(msg)

    geo = geo if in_place else geo.model_copy(deep=True)
    i, j = idxs

    # Compute current distance and unit vector
    vec = geo.coordinates[j] - geo.coordinates[i]
    r = np.linalg.norm(vec)
    unit_vec = vec / r

    # Ensure that change does not exceed max allowable
    # NOTE: Can be replaced by structure smoothing / verification
    dr = abs(r - val)
    if dr > max_change:
        msg = f"{dr = } exceeds {max_change = }."
        raise ValueError(msg)

    # Atom j coordinates relevant to atom i
    geo.coordinates[j] = geo.coordinates[i] + (unit_vec * val)

    return geo
