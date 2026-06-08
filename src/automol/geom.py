"""Molecular geometry functions."""

import itertools
from collections.abc import Collection, Sequence

import numpy as np
import pint
import scipy
from automatics import Geometry, Identity, element
from automatics.types import FloatArray
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

RADIANS_TO_DEGREES = pint.Quantity("radian").m_as("degree")
DEGREES_TO_RADIANS = 1 / RADIANS_TO_DEGREES


def center_of_mass(geo: Geometry) -> FloatArray:
    """
    Calculate geometry center of mass.

    Parameters
    ----------
        Geometry.

    Returns
    -------
        Center of mass coordinates.
    """
    masses = list(map(element.mass, geo.symbols))
    coords = geo.coordinates
    return np.sum(np.reshape(masses, (-1, 1)) * coords, axis=0) / np.sum(masses)


def distance_matrix(geo: Geometry) -> FloatArray:
    """
    Compute the distance matrix for a geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    FloatArray
        Distance matrix of geometry.
    """
    return scipy.spatial.distance_matrix(geo.coordinates, geo.coordinates)


def dihedral_angle(
    geo: Geometry, keys: Sequence[int], *, degrees: bool = True
) -> float:
    """Calculate the dihedral angle defined by four atoms.

    Parameters
    ----------
    geo
        Geometry.
    keys
        Indices of the four atoms defining the dihedral angle.
    degrees
        Whether to return the angle in degrees or radians.

    Returns
    -------
        Dihedral angle.
    """
    coords = geo.coordinates[list(keys)]
    if len(coords) != 4:  # noqa: PLR2004
        msg = "Exactly four atoms must be specified for dihedral angle."
        raise ValueError(msg)

    # Determine bond vectors and 1-2-3 plane normal
    r1, r2, r3, r4 = coords
    r12 = r2 - r1
    r23 = r3 - r2
    r34 = r4 - r3
    n123 = np.cross(r12, r23)

    # Form coordinate system with x upward in plane, y along plane normal, and z
    # away along central bond:
    #
    #     x
    #     ^
    #     1
    #     |
    #     2/3   > y
    #      \
    #       4
    #
    z = r23 / np.linalg.norm(r23)
    y = n123 / np.linalg.norm(n123)
    x = np.cross(y, z)

    # Determine components of 3-4 bond along x and y and calculate angle from arctan
    v = r34 / np.linalg.norm(r34)
    vx = np.dot(v, x)
    vy = np.dot(v, y)
    angle = np.arctan2(vy, vx)
    return angle * RADIANS_TO_DEGREES if degrees else angle


def inertia_tensor(geo: Geometry) -> FloatArray:
    """Calculate the inertia tensor of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia tensor.
    """
    masses = geo.masses
    coords = geo.coordinates - center_of_mass(geo)
    return sum(
        m * (np.vdot(r, r) * np.eye(3) - np.outer(r, r))
        for (r, m) in zip(coords, masses, strict=True)
    )


def inertia_moments(geo: Geometry) -> FloatArray:
    """Calculate the moments of inertia of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia moments.
    """
    evals, *_ = rotational_analysis(geo)
    return evals


def inertia_axes(geo: Geometry) -> FloatArray:
    """Calculate the axes of inertia of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia axes.
    """
    _, evecs = rotational_analysis(geo)
    return evecs


def rotation_to_inertia_axes(geo: Geometry) -> Rotation:
    """Return a rotation that aligns the geometry with its principal axes.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Rotation object.
    """
    evecs = inertia_axes(geo)
    return Rotation.from_matrix(evecs.T)


def rotational_analysis(geo: Geometry) -> tuple[FloatArray, FloatArray]:
    """Calculate rotational analysis of a geometry.

    Parameters
    ----------
    geo
        Geometry.
    drop_null
        Whether to drop null eigenvalues.

    Returns
    -------
        Eigenvalues and eigenvectors of the inertia tensor.
    """
    inert = inertia_tensor(geo)
    evals, evecs = np.linalg.eigh(inert)
    # Ensure right-handed coordinate system
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1  # flip one eigenvector
    return evals, evecs


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
    centroid_p = center_of_mass(geo1)
    centroid_q = center_of_mass(geo2)
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


def is_similar(geo1: Geometry, geo2: Geometry) -> bool:
    """
    Determine whether two geometries are similar.

    Parameters
    ----------
    geo1
        Geometry object.
    geo2
        Geometry object.

    Returns
    -------
    bool
        Whether the two geometries are similar.
    """
    # --- Geometry Hash ---
    if geo1.hash == geo2.hash:
        return True

    # --- InChI    ---
    inchi1 = Identity.from_geometry(geo1, algorithm="rdkit inchi")
    inchi2 = Identity.from_geometry(geo1, algorithm="rdkit inchi")

    if inchi1.value != inchi2.value:
        return False

    # --- Heavy Atom RMSD ---
    if geo1.symbols != geo2.symbols:
        msg = "Atomic symbols do not map onto each other. RMSD cannot be computed."
        raise ValueError(msg)

    msg = "Not implemented until canonical ordering is established."
    raise NotImplementedError(msg)


# Transformation
def translate(
    geo: Geometry,
    arr: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Translate geometry.

    Parameters
    ----------
    geo
        Geometry.
    arr
        Translation vector or matrix.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = np.add(geo.coordinates[mask], arr)
    return geo


def reflect(
    geo: Geometry,
    normal: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Reflect geometry across a plane.

    Parameters
    ----------
    geo
        Geometry.
    normal
        Normal vector of the reflection plane.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    normal = np.asarray(normal, dtype=float)
    proj = np.outer(normal, normal) / np.dot(normal, normal)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = geo.coordinates[mask] - 2 * geo.coordinates[mask] @ proj
    return geo


def rotate(
    geo: Geometry,
    rot: Rotation,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Rotate geometry.

    Parameters
    ----------
    geo
        Geometry.
    rot
        Rotation object.
    keys
        Atoms to rotate. If None, rotate all atoms.
    in_place
        Whether to rotate in place or return a new geometry.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = rot.apply(geo.coordinates[mask])
    return geo


def to_eckart_frame(geo: Geometry, *, in_place: bool = False) -> Geometry:
    """Rotate geometry to align with inertia axes.

    Parameters
    ----------
    geo
        Geometry.
    in_place
        Whether to rotate in place or return a new geometry.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    # Move to center of mass
    geo = translate(geo, -center_of_mass(geo), in_place=True)
    # Rotate to inertia axes
    rot = rotation_to_inertia_axes(geo)
    return rotate(geo, rot, in_place=True)


def set_distance(
    geo: Geometry,
    *,
    idxs: Sequence[int],
    val: float,
    max_change: float = 0.25,
    in_place: bool = False,
) -> Geometry:
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


def adjacency_matrix(
    geo: Geometry,
    *,
    delta: float = 0.5,
    override_bonding_capacities: dict[str, int] | None = None,
) -> ArrayLike:
    """Determine neighboring atoms."""
    dmat = distance_matrix(geo)
    radii = np.array(geo.covalent_radii)

    r_cov_matrix = radii[:, np.newaxis] + radii[np.newaxis, :] + delta
    amat = dmat <= r_cov_matrix
    np.fill_diagonal(amat, 0)  # Atoms don't neighbor themselves

    caps = [
        element.bonding_capacity(s, override=override_bonding_capacities)
        for s in geo.symbols
    ]
    vals = np.sum(amat, axis=0)

    for i, (symb, val, cap) in enumerate(zip(geo.symbols, vals, caps, strict=True)):
        if val > cap:
            msg = f"Atom {symb}:{i} degree ({val}) exceeds bonding capacity ({cap})."
            raise NotImplementedError(msg)

    return amat
