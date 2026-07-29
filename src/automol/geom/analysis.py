"""Geometric, structural, and vibrational analysis of a geometry."""

from collections.abc import Sequence
from itertools import permutations
from math import factorial
from typing import TYPE_CHECKING, Literal

import irmsd
import numpy as np
import numpy.typing as npt
from pynauty import Graph, autgrp, canon_label
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from ..utils import constants, element
from ..utils.types import FloatArray

if TYPE_CHECKING:
    from .core import Geometry

RMSD_THRESHOLD = 0.125

# Bound on n_perms * k for the brute-force orbit assignment in
# `_assign_orbit`; past this, a Hungarian solve is cheaper than enumerating k!.
_MAX_BRUTE_FORCE_ELEMENTS = 200_000


# Properties
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
    geo: "Geometry",
    *,
    sigma: float = 1.3,
    flood_fill: bool = False,
    enforce_valence: bool = False,
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
    enforce_valence
        If True, enforce maximum valence constraints by keeping only the closest
        neighbors up to each atom's maximum valence.

    Returns
    -------
    np.ndarray
        2D binary adjacency matrix.
    """
    radii = np.array(geo.covalent_radii)
    max_valence = np.array(geo.valences)
    dmat = distance_matrix(geo)

    while True:
        umat = sigma * (radii[:, None] + radii[None, :])
        amat = (dmat < umat).astype(int)
        np.fill_diagonal(amat, 0)

        if enforce_valence:
            for i in range(len(amat)):
                bond_count = np.sum(amat[i])
                if bond_count > max_valence[i]:
                    neighbors = np.where(amat[i] > 0)[0]
                    distances = dmat[i, neighbors]
                    closest_idx = np.argsort(distances)[: max_valence[i]]
                    closest_neighbors = neighbors[closest_idx]
                    excess = np.setdiff1d(neighbors, closest_neighbors)
                    for j in excess:
                        amat[i, j] = 0
                        amat[j, i] = 0

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


# Internal coordinates
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


# Inertia and rotational analysis
def inertia_tensor(geo: "Geometry") -> np.ndarray:
    """Calculate the inertia tensor of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia tensor.

    Example
    -------
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
    masses = geo.masses
    coords = geo.coordinates - center_of_mass(geo)
    return sum(
        m * (np.vdot(r, r) * np.eye(3) - np.outer(r, r))
        for (r, m) in zip(coords, masses, strict=True)
    )


def inertia_moments(geo: "Geometry") -> np.ndarray:
    """Calculate the moments of inertia of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia moments.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> moments = inertia_moments(geo)
    >>> moments.shape
    (3,)
    >>> bool(np.all(moments >= 0))
    True
    """
    evals, *_ = rotational_analysis(geo)
    return evals


def inertia_axes(geo: "Geometry") -> np.ndarray:
    """Calculate the axes of inertia of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia axes.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> axes = inertia_axes(geo)
    >>> bool(np.allclose(axes.T @ axes, np.eye(3)))
    True
    """
    _, evecs = rotational_analysis(geo)
    return evecs


def rotational_analysis(geo: "Geometry") -> tuple[np.ndarray, np.ndarray]:
    """Calculate rotational analysis of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Eigenvalues and eigenvectors of the inertia tensor.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> evals, evecs = rotational_analysis(geo)
    >>> evals.shape, evecs.shape
    ((3,), (3, 3))
    >>> bool(np.linalg.det(evecs) > 0)
    True
    """
    inert = inertia_tensor(geo)
    evals, evecs = np.linalg.eigh(inert)
    # Ensure right-handed coordinate system
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1  # flip one eigenvector
    return evals, evecs


def rotation_to_inertia_axes(geo: "Geometry") -> Rotation:
    """Return a rotation that aligns the geometry with its principal axes.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Rotation object.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> isinstance(rotation_to_inertia_axes(geo), Rotation)
    True
    """
    evecs = inertia_axes(geo)
    return Rotation.from_matrix(evecs.T)


# Vibrational normal-mode and frequency analysis
def mass_weight_vector(geo: "Geometry") -> np.ndarray:
    """Get the mass-weighting vector of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry

    Returns
    -------
        Mass-weighting vector.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> mass_weight_vector(geo).shape
    (9,)
    """
    return np.sqrt(np.repeat(geo.masses, 3))


def translational_normal_modes(
    geo: "Geometry", *, mass_weighted: bool = True
) -> np.ndarray:
    """Calculate translational normal modes of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry
    mass_weighted
        If True, return mass-weighted normal modes.

    Returns
    -------
        Translational normal modes.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> translational_normal_modes(geo).shape
    (9, 3)
    """
    trans_coos = np.tile(np.eye(3), (geo.atom_count, 1))

    if mass_weighted:
        trans_coos *= np.sqrt(mass_weight_vector(geo))[:, np.newaxis]

    return trans_coos


def rotational_normal_modes(
    geo: "Geometry", *, mass_weighted: bool = True
) -> np.ndarray:
    """Calculate rotational normal modes of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry
    mass_weighted
        If True, return mass-weighted normal modes.

    Returns
    -------
        Rotational normal modes.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> rotational_normal_modes(geo).shape
    (9, 3)
    """
    _, rot_axes = rotational_analysis(geo)
    coords = (geo.coordinates - center_of_mass(geo)) * constants.ANGSTROM_TO_BOHR
    rot_coos = [
        np.concatenate([np.cross(xyz, rot_axis) for xyz in coords])
        for rot_axis in rot_axes
    ]
    rot_coos = np.transpose(rot_coos)

    if mass_weighted:
        rot_coos *= np.sqrt(mass_weight_vector(geo))[:, np.newaxis]

    return rot_coos


def normal_mode_projection(
    geo: "Geometry", *, trans: bool = False, rot: bool = False
) -> np.ndarray:
    """Get the matrix for projecting onto a subset of normal modes.

    Parameters
    ----------
    geo
        Instance of a Geometry.
    trans
        If True, keep translational modes.
    rot
        If True, keep rotational modes.

    Returns
    -------
        Normal mode projection matrix.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> normal_mode_projection(geo).shape
    (9, 3)
    >>> normal_mode_projection(geo, trans=True, rot=True).shape
    (9, 9)
    """
    coos = []
    if not trans:
        coos.append(translational_normal_modes(geo, mass_weighted=True))

    if not rot:
        coos.append(rotational_normal_modes(geo, mass_weighted=True))

    if not coos:
        return np.eye(geo.atom_count * 3)

    coos = np.hstack(coos)
    dim = np.shape(coos)[-1]
    coo_basis, *_ = np.linalg.svd(coos, full_matrices=True)

    return coo_basis[:, dim:]


def vibrational_analysis(
    geo: "Geometry", hess: np.ndarray, *, trans: bool = False, rot: bool = False
) -> tuple[tuple[float, ...], npt.NDArray]:
    """Calculate frequencies and vibrational modes from a Hessian matrix.

    Parameters
    ----------
    geo
        Instance of a Geometry
    hess
        `(3N, 3N)` array of hessian values.
    trans
        If `True`, keep translational modes.
    rot
        If `True`, keep rotational modes.

    Returns
    -------
    frequencies
        Vibrational frequencies.
    modes
        Vibrational modes.

    Raises
    ------
    ValueError
        `Hessian shape is not (3N, 3N)`.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> freqs, modes = vibrational_analysis(geo, np.eye(9))
    >>> len(freqs)
    3
    >>> modes.shape
    (9, 3)
    """
    exp_dim = 3 * geo.atom_count
    if len(hess) != exp_dim or len(hess[0]) != exp_dim:
        msg = f"Hessian shape ({len(hess)}, {len(hess[0])}) is not (3N, 3N)."
        raise ValueError(msg)

    masses = mass_weight_vector(geo)
    hess_mw = hess / np.outer(masses, masses)

    proj = normal_mode_projection(geo, trans=trans, rot=rot)
    hess_proj = proj.T @ hess_mw @ proj

    evals, evecs = np.linalg.eigh(hess_proj)

    norm_coos = np.dot(proj, evecs) / masses[:, np.newaxis]
    norm_coos /= np.linalg.norm(norm_coos, axis=0)

    freqs = (
        np.sqrt(np.complex128(evals)) * constants.VIBRATIONAL_FORCE_TO_INV_CM_FREQUENCY
    )
    freqs = tuple(map(float, np.real(freqs) - np.imag(freqs)))

    return freqs, norm_coos


def harmonic_zpv(
    geo: "Geometry", hess: list[list[float]], *, freqs: tuple[float, ...] | None = None
) -> float:
    """Calculate the harmonic zero point vibrational energy of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry.
    hess
        `(3N, 3N)` array of hessian values. Ignored if `freqs` is given.
    freqs
        Precomputed vibrational frequencies (in cm^-1). If given, `hess` is
        not used and `vibrational_analysis` is skipped.

    Returns
    -------
        Harmonic zero point vibrational energy, in Hartree.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> zpe = harmonic_zpv(geo, hess=[], freqs=(1000.0, 2000.0, -50.0))
    >>> round(zpe, 6)
    0.006835
    """
    if freqs is None:
        freqs, _ = vibrational_analysis(geo, np.array(hess))

    zpe_wavenumbers = 0.5 * sum([f for f in freqs if f > 0.0])

    return zpe_wavenumbers * constants.WAVENUMBER_TO_HARTREE


# Comparison
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


def bond_graph(geo: "Geometry") -> Graph:
    """Build a pynauty graph representing geo's bond connectivity.

    Parameters
    ----------
    geo
        Geometry to build a bond graph for.

    Returns
    -------
    Graph
        Pynauty graph over geo's atoms, in geo's own atom order (not
        canonically relabeled). Use `canon_label` on this graph to compute
        the permutation into canonical order.
    """
    amat = adjacency_matrix(geo, flood_fill=True, enforce_valence=True)
    adjacency = {i: list(np.where(amat[i])[0]) for i in range(len(amat))}
    return Graph(geo.atom_count, adjacency_dict=adjacency)


def orbit_classes(geo: "Geometry") -> list[np.ndarray]:
    """Group geo's atom indices by bond-graph automorphism orbit.

    Atoms sharing an orbit are graph-equivalent (e.g. the three H's of a
    methyl group) and may be freely reassigned among each other's positions
    when matching two conformers.

    Parameters
    ----------
    geo
        Geometry to compute orbits for (in its own atom order).

    Returns
    -------
    list[np.ndarray]
        One array of atom indices per orbit.
    """
    _, _, _, orbits, _ = autgrp(bond_graph(geo))
    classes: dict[int, list[int]] = {}
    for atom_idx, orbit_id in enumerate(orbits):
        classes.setdefault(orbit_id, []).append(atom_idx)
    return [np.array(idxs) for idxs in classes.values()]


def kabsch_align(ref_coords: np.ndarray, tgt_coords: np.ndarray) -> np.ndarray:
    """Rigidly rotate tgt_coords onto ref_coords (centered).

    Parameters
    ----------
    ref_coords
        Reference coordinates, shape (n_atoms, 3).
    tgt_coords
        Target coordinates, shape (n_atoms, 3).

    Returns
    -------
    np.ndarray
        tgt_coords, centered and rotated to best fit ref_coords (also
        centered), shape (n_atoms, 3).
    """
    ref_ctr = ref_coords - ref_coords.mean(axis=0)
    tgt_ctr = tgt_coords - tgt_coords.mean(axis=0)

    cov = tgt_ctr.T @ ref_ctr
    u, _, vt = np.linalg.svd(cov)
    rot = u @ vt

    # A negative determinant means this "rotation" is actually a reflection;
    # flip the least-significant singular vector to force a proper rotation.
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = u @ vt

    return tgt_ctr @ rot


def _assign_orbit(
    aligned_cls: np.ndarray, ref_cls: np.ndarray, perm_cls: np.ndarray
) -> np.ndarray:
    """Solve the optimal within-orbit re-assignment.

    Squared-distance assignment cost decomposes as
    |a_i|^2 + |r_j|^2 - 2 a_i.r_j, and the assignment problem is invariant to
    constants added to a full row or column, so the |a_i|^2 (row) and |r_j|^2
    (column) terms can't affect the optimal assignment. This reduces the
    problem to maximizing sum_j a_p(j).r_j over permutations p, via one dot
    product (aligned_cls @ ref_cls.T).

    For small orbits (the common case: methyls, geminal pairs, ...) every
    permutation is enumerated and scored in one vectorized pass. Larger
    orbits fall back to a Hungarian solve.

    Parameters
    ----------
    aligned_cls
        Aligned target coordinates within the orbit, shape (k, 3).
    ref_cls
        Reference coordinates within the orbit, shape (k, 3).
    perm_cls
        Current atom-index correspondence within the orbit, shape (k,).

    Returns
    -------
    np.ndarray
        Updated atom-index correspondence within the orbit, shape (k,).
    """
    k = aligned_cls.shape[0]
    dot = aligned_cls @ ref_cls.T  # (k, k): dot[i, j] = a_i . r_j

    n_perms = factorial(k)
    if n_perms * k <= _MAX_BRUTE_FORCE_ELEMENTS:
        perms = np.array(list(permutations(range(k))))  # (n_perms, k)
        # scores[p] = sum_j dot[perms[p, j], j]
        scores = dot[perms, np.arange(k)].sum(axis=1)
        best = perms[np.argmax(scores)]
    else:
        _, col_idx = linear_sum_assignment(-dot)
        best = np.argsort(col_idx)

    return perm_cls[best]


def _hungarian_correspondence(
    ref_coords: np.ndarray,
    tgt_coords: np.ndarray,
    classes: list[np.ndarray],
    init_perm: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Align (Kabsch) and reassign (Hungarian) iteratively to a local optimum.

    ref_coords and tgt_coords are shape (n_atoms, 3); init_perm is shape
    (n_atoms,).

    Returns
    -------
    np.ndarray
        Converged atom-index correspondence, same shape as init_perm.
    """
    ref_ctr = ref_coords - ref_coords.mean(axis=0)

    # perm[i] = index of the target atom currently assigned to ref position i
    perm = init_perm
    for _ in range(max_iter):
        gathered = tgt_coords[perm]
        aligned_tgt = kabsch_align(ref_coords, gathered)

        new_perm = perm.copy()
        changed = False
        for cls in classes:
            if len(cls) == 1:
                continue
            new_sub = _assign_orbit(aligned_tgt[cls], ref_ctr[cls], perm[cls])
            if not np.array_equal(new_sub, perm[cls]):
                changed = True
            new_perm[cls] = new_sub

        perm = new_perm
        if not changed:
            break

    return perm


def hungarian_correspondence(
    ref_geo: "Geometry",
    tgt_geo: "Geometry",
    *,
    classes: list[np.ndarray] | None = None,
    max_iter: int = 10,
) -> np.ndarray:
    """Find the atom-index correspondence between two geometries via Hungarian solve.

    Resolves symmetric atom correspondence using only the bond-graph
    automorphism orbit partition (from `orbit_classes`, itself derived from
    nauty's `autgrp` at no extra cost) instead of enumerating the full
    automorphism group, which blows up combinatorially for molecules with
    many equivalent atoms. Iterates between (1) rigidly aligning tgt_geo onto
    ref_geo via Kabsch (`kabsch_align`) using the current correspondence and
    (2) re-solving the optimal correspondence within each orbit via the
    Hungarian algorithm, scaling polynomially instead of factorially.

    ref_geo and tgt_geo must have the same connectivity and already be in a
    consistent atom order (e.g. both canonically relabeled, see `bond_graph`
    and `canon_label`) -- this function only permutes atoms *within* orbits,
    so it cannot fix up an otherwise-mismatched atom order on its own.

    Parameters
    ----------
    ref_geo
        Reference geometry.
    tgt_geo
        Target geometry to find a correspondence for.
    classes
        Precomputed orbit classes for ref_geo (from `orbit_classes`). If not
        given, they are computed from ref_geo's bond graph.
    max_iter
        Maximum number of align/reassign iterations.

    Returns
    -------
    np.ndarray
        Indices into tgt_geo's atoms such that `tgt_geo.relabel_atoms(result)`
        best corresponds to ref_geo, shape (n_atoms,).
    """
    if classes is None:
        classes = orbit_classes(ref_geo)

    ref_coords = np.asarray(ref_geo.coordinates)
    tgt_coords = np.asarray(tgt_geo.coordinates)
    init_perm = np.arange(ref_coords.shape[0])

    return _hungarian_correspondence(
        ref_coords, tgt_coords, classes, init_perm, max_iter
    )


def _rmsd_from_correspondence(
    ref_coords: np.ndarray, tgt_coords: np.ndarray, perm: np.ndarray
) -> float:
    """Compute RMSD after Kabsch-aligning tgt_coords[perm] onto ref_coords."""
    ref_ctr = ref_coords - ref_coords.mean(axis=0)
    aligned_tgt = kabsch_align(ref_coords, tgt_coords[perm])
    return float(np.sqrt(np.mean(np.sum((aligned_tgt - ref_ctr) ** 2, axis=1))))


def assignment_rmsd(
    ref_geo: "Geometry",
    tgt_geo: "Geometry",
    max_iter: int = 10,
    n_restarts: int = 100,
    seed: int = 0,
) -> float:
    """RMSD between ref_geo and tgt_geo via iterative Hungarian assignment.

    Finds the atom correspondence via `hungarian_correspondence`, then
    reports the Kabsch-aligned RMSD at that correspondence.

    Since the underlying local search (see `hungarian_correspondence`) can
    settle into a local optimum depending on its starting correspondence, it
    is repeated from several randomized within-orbit starting correspondences
    (plus the identity correspondence), and the best result is kept -- the
    standard multi-start ICP fix.

    Parameters
    ----------
    ref_geo
        Reference geometry.
    tgt_geo
        Target geometry to align (must have the same connectivity as ref_geo).
    max_iter
        Maximum number of align/reassign iterations per restart.
    n_restarts
        Number of randomized starting correspondences to try, in addition to
        the identity correspondence.
    seed
        Seed for the random restarts, for reproducibility.

    Returns
    -------
    float
        Best-fit RMSD found between ref_geo and tgt_geo across all restarts.
    """
    can_ref = ref_geo.relabel_atoms(canon_label(bond_graph(ref_geo)))
    can_tgt = tgt_geo.relabel_atoms(canon_label(bond_graph(tgt_geo)))
    classes = orbit_classes(can_ref)

    ref_coords = np.asarray(can_ref.coordinates)
    tgt_coords = np.asarray(can_tgt.coordinates)
    n_atoms = tgt_coords.shape[0]

    rng = np.random.default_rng(seed)
    identity_perm = np.arange(n_atoms)
    perm = _hungarian_correspondence(
        ref_coords, tgt_coords, classes, identity_perm, max_iter
    )
    best_rmsd = _rmsd_from_correspondence(ref_coords, tgt_coords, perm)

    for _ in range(n_restarts):
        init_perm = identity_perm.copy()
        for cls in classes:
            init_perm[cls] = rng.permutation(init_perm[cls])
        perm = _hungarian_correspondence(
            ref_coords, tgt_coords, classes, init_perm, max_iter
        )
        rmsd = _rmsd_from_correspondence(ref_coords, tgt_coords, perm)
        best_rmsd = min(best_rmsd, rmsd)

    return best_rmsd
