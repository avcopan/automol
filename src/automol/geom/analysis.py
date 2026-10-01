"""Geometric and structural analysis of a geometry."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy import spatial
from scipy.sparse.csgraph import connected_components

from .. import rd
from ..utils.types import FloatArray

if TYPE_CHECKING:
    from .core import Geometry

FLOOD_FILL_STEP = 0.05


# Properties
def center_of_mass(geo: "Geometry") -> FloatArray:
    """Calculate the geometry center of mass.

    Args:
        geo: Geometry.

    Returns:
        FloatArray: Center of mass coordinates (the origin for an empty
        geometry).
    """
    masses = np.asarray(geo.masses)
    if not masses.size:
        return np.zeros(3)
    return masses @ geo.coordinates / np.sum(masses)


def distance_matrix(geo: "Geometry") -> np.ndarray:
    """Calculate the geometry distance matrix.

    Args:
        geo: Geometry.

    Returns:
        np.ndarray: Distance matrix of the geometry.
    """
    return spatial.distance.cdist(geo.coordinates, geo.coordinates)


def adjacency_matrix(
    geo: "Geometry",
    *,
    sigma: float = 1.3,
    flood_fill: bool = False,
    enforce_valence: bool = False,
) -> np.ndarray:
    """Compute the molecular adjacency matrix based on covalent radii.

    An edge exists between two atoms if their distance is less than `sigma`
    times the sum of their covalent radii.

    Args:
        geo: Geometry.
        sigma: Scaling factor applied to the sum of covalent radii.
        flood_fill: If `True`, increase `sigma` in steps of `FLOOD_FILL_STEP`
            until the adjacency matrix is connected (one fragment). Stops early if
            increasing `sigma` can no longer add edges, in which case the result
            may remain disconnected.
        enforce_valence: If `True`, cap the number of bonds to each atom at its
            maximum valence (see `automol.rd.mol.max_bond_count`). Candidate bonds
            are accepted in order of increasing distance relative to the sum of
            covalent radii, so the result does not depend on atom ordering.

    Returns:
        np.ndarray: 2D binary adjacency matrix.
    """
    natms = geo.atom_count
    radii = np.asarray(geo.covalent_radii)
    # Pair distance relative to the sum of covalent radii; bonded if below sigma
    iu, ju = np.triu_indices(natms, k=1)
    ratios = distance_matrix(geo)[iu, ju] / (radii[iu] + radii[ju])
    order = np.argsort(ratios, kind="stable")
    iu, ju, ratios = iu[order], ju[order], ratios[order]

    max_bonds = (
        [rd.mol.max_bond_count(s) for s in geo.symbols] if enforce_valence else None
    )

    while True:
        amat = np.zeros((natms, natms), dtype=int)
        ncand = int(np.searchsorted(ratios, sigma, side="left"))
        if max_bonds is None:
            amat[iu[:ncand], ju[:ncand]] = amat[ju[:ncand], iu[:ncand]] = 1
        else:
            counts = np.zeros(natms, dtype=int)
            for i, j in zip(iu[:ncand], ju[:ncand], strict=True):
                if counts[i] < max_bonds[i] and counts[j] < max_bonds[j]:
                    amat[i, j] = amat[j, i] = 1
                    counts[[i, j]] += 1

        if not flood_fill or ncand == len(ratios):
            break

        n_components, _ = connected_components(amat, directed=False)
        if n_components <= 1:
            break
        sigma += FLOOD_FILL_STEP

    return amat


def distance_keys(geo: "Geometry") -> np.ndarray:
    """Generate a sorted distance descriptor.

    The descriptor characterizes a geometry using the distance between all unique
    atom pairs, sorted first by atomic numbers (z_low, z_high) then their pairwise
    distances.

    Args:
        geo: Geometry.

    Returns:
        np.ndarray: Array where each row contains `[z1, z2, distance]`.
    """
    z = np.asarray(geo.atomic_numbers)
    dmat = distance_matrix(geo)
    iu, ju = np.triu_indices(geo.atom_count, k=1)

    pairs = np.sort(np.stack([z[iu], z[ju]], axis=1), axis=1)
    key = np.column_stack([pairs.astype(float), dmat[iu, ju]])

    return _sort_rows(key)


# Internal coordinates
def bonds(geo: "Geometry", amat: ArrayLike) -> np.ndarray:
    """Compile distances of bonded pairs.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        np.ndarray: Sorted array of bonded pairs and their distances
        `[[z1, z2, dist], ...]`, with `z1 <= z2`. Shape is ``(nbonds, 3)``.

    Example:
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
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    i, j = np.nonzero(np.triu(_as_adjacency(amat, geo.atom_count)))

    dist = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    rows = np.column_stack([np.minimum(zs[i], zs[j]), np.maximum(zs[i], zs[j]), dist])
    return _sort_rows(rows.reshape(-1, 3))


def angles(geo: "Geometry", amat: ArrayLike) -> np.ndarray:
    """Compile angles of bonded triples.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        np.ndarray: Sorted array of bonded triples and their angles in radians
        `[[z1, z2, z3, theta], ...]`, where `z2` is the central atom and
        `z1 <= z3`. Shape is ``(nangles, 4)``.

    Example:
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
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    nbrs = _neighbors(_as_adjacency(amat, geo.atom_count))

    triples = [
        (i, j, k)
        for j, nbr in enumerate(nbrs)
        for pos, i in enumerate(nbr)
        for k in nbr[pos + 1 :]
    ]
    i, j, k = np.array(triples, dtype=int).reshape(-1, 3).T

    v1 = xyz[i] - xyz[j]
    v2 = xyz[k] - xyz[j]
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(np.einsum("nt,nt->n", v1, v2) / norms, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    rows = np.column_stack(
        [np.minimum(zs[i], zs[k]), zs[j], np.maximum(zs[i], zs[k]), theta]
    )
    return _sort_rows(rows.reshape(-1, 4))


def dihedrals(geo: "Geometry", amat: ArrayLike) -> np.ndarray:
    """Compile dihedrals of bonded quadruples.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        np.ndarray: Sorted array of bonded quadruples and their dihedral angles
        in radians `[[z1, z2, z3, z4, phi], ...]`. Each quadruple is oriented so
        that `(z1, z2, z3, z4) <= (z4, z3, z2, z1)`, which leaves `phi`
        unchanged. Shape is ``(ndihedrals, 5)``.

    Example:
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
        >>> [float(x) for x in d[0][:4]]
        [1.0, 8.0, 8.0, 1.0]
        >>> round(float(d[0][-1]), 6)
        1.570796
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    amat = _as_adjacency(amat, geo.atom_count)
    nbrs = _neighbors(amat)

    # Each dihedral is enumerated once, about its unordered central bond (j, k)
    quads = [
        (i, j, k, l)
        for j, k in zip(*np.nonzero(np.triu(amat)), strict=True)
        for i in nbrs[j]
        if i != k
        for l in nbrs[k]  # noqa: E741
        if l not in (i, j)
    ]
    i, j, k, l = np.array(quads, dtype=int).reshape(-1, 4).T  # noqa: E741

    b1 = xyz[j] - xyz[i]
    b2 = xyz[k] - xyz[j]
    b3 = xyz[l] - xyz[k]

    nb2 = np.linalg.norm(b2, axis=1, keepdims=True)
    nb2 = np.where(nb2 == 0.0, 1.0, nb2)
    ub2 = b2 / nb2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    x = np.einsum("nt,nt->n", n1, n2)
    y = np.einsum("nt,nt->n", np.cross(n1, n2), ub2)
    phi = np.arctan2(y, x)

    zquads = zs[np.column_stack([i, j, k, l]).astype(int)].reshape(-1, 4)
    zquads = np.array(
        [min(tuple(zq), tuple(zq[::-1])) for zq in zquads.tolist()], dtype=float
    ).reshape(-1, 4)
    return _sort_rows(np.column_stack([zquads, phi]))


# Inertia and rotational analysis
def inertia_tensor(geo: "Geometry") -> np.ndarray:
    """Calculate the inertia tensor of a geometry.

    Args:
        geo: Geometry.

    Returns:
        np.ndarray: Inertia tensor (zero for an empty geometry).

    Example:
        >>> from automol import Geometry
        >>> geo = Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> tensor = inertia_tensor(geo)
        >>> tensor.shape
        (3, 3)
        >>> bool(np.allclose(tensor, tensor.T))
        True
    """
    masses = np.asarray(geo.masses)
    coords = geo.coordinates - center_of_mass(geo)
    mr2 = np.einsum("i,ij,ij->", masses, coords, coords)
    return mr2 * np.eye(3) - np.einsum("i,ij,ik->jk", masses, coords, coords)


# Helpers
def _as_adjacency(amat: ArrayLike, natms: int) -> np.ndarray:
    """Convert an adjacency matrix to a validated boolean array.

    Args:
        amat: Adjacency matrix-like object.
        natms: Expected number of atoms.

    Returns:
        np.ndarray: Boolean adjacency matrix with shape ``(natms, natms)``.

    Raises:
        ValueError: If `amat` does not have shape ``(natms, natms)``.
    """
    amat = np.asarray(amat).astype(bool)
    if amat.shape != (natms, natms):
        msg = f"Expected adjacency matrix of shape {(natms, natms)}, got {amat.shape}."
        raise ValueError(msg)
    return amat


def _neighbors(amat: np.ndarray) -> list[list[int]]:
    """Get sorted neighbor lists from an adjacency matrix.

    Args:
        amat: Boolean adjacency matrix.

    Returns:
        list[list[int]]: Neighbor indices for each atom.
    """
    return [np.flatnonzero(row).tolist() for row in amat]


def _sort_rows(arr: np.ndarray) -> np.ndarray:
    """Sort the rows of a 2D array lexicographically.

    Args:
        arr: Two-dimensional array to sort.

    Returns:
        np.ndarray: Lexicographically sorted array.
    """
    return arr[np.lexsort(arr.T[::-1])]
