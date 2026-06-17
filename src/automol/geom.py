"""Molecular geometry functions."""

import itertools
from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt
from automatics import Geometry, element
from automatics.utils import constants
from automatics.utils.types import FloatArray
from numpy.typing import ArrayLike
from scipy import spatial
from scipy.spatial.transform import Rotation


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
    return spatial.distance_matrix(geo.coordinates, geo.coordinates)


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

    z = r23 / np.linalg.norm(r23)
    y = n123 / np.linalg.norm(n123)
    x = np.cross(y, z)

    # Determine components of 3-4 bond along x and y and calculate angle from arctan
    v = r34 / np.linalg.norm(r34)
    vx = np.dot(v, x)
    vy = np.dot(v, y)
    angle = np.arctan2(vy, vx)
    return angle * constants.RADIANS_TO_DEGREES if degrees else angle


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


def mass_weight_vector(geo: Geometry) -> np.ndarray:
    """Get the mass-weighting vector of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry

    Returns
    -------
    mass-weighting vector
    """
    return np.sqrt(np.repeat(geo.masses, 3))


def rotational_analysis(geo: Geometry) -> tuple[FloatArray, FloatArray]:
    """Calculate rotational analysis of a geometry.

    Parameters
    ----------
    geo
        Geometry.

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


def translational_normal_modes(
    geo: Geometry, *, mass_weighted: bool = True
) -> np.ndarray:
    """Calculate translational normal modes of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry
    mass_weighted
        If True, return mass-weighted normal modes.
    """
    trans_coos = np.tile(np.eye(3), (geo.atom_count, 1))

    if mass_weighted:
        trans_coos *= np.sqrt(mass_weight_vector(geo))[:, np.newaxis]

    return trans_coos


def rotational_normal_modes(geo: Geometry, *, mass_weighted: bool = True) -> np.ndarray:
    """Calculate rotational normal modes of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry
    mass_weighted
        If True, return mass-weighted normal modes.
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
    geo: Geometry, *, trans: bool = False, rot: bool = False
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
    geo: Geometry, hess: list[list[float]], *, trans: bool = False, rot: bool = False
) -> tuple[tuple[float, ...], npt.NDArray]:
    """Calculate frequencies and vibrational modes from a Hessian matrix.

    Parameters
    ----------
    geo
        Instance of a Geometry
    hess
        `(3N, 3N)` array of hessian values.
    trans
        If True, keep translational modes.
    rot
        If True, keep rotational modes.

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

    freqs = np.sqrt(np.complex128(evals)) * constants.AU_TO_INV_CM
    freqs = tuple(map(float, np.real(freqs) - np.imag(freqs)))

    return freqs, norm_coos


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
