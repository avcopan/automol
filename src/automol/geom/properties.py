"""Molecular geometry functions."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import spatial

from .. import element
from ..utils import constants
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


def distance_matrix(geo: "Geometry") -> FloatArray:
    """Calculate the geometry distance matrix.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
    FloatArray
        Distance matrix of geometry.
    """
    return spatial.distance_matrix(geo.coordinates, geo.coordinates)


def adjacency_matrix(
    geo: "Geometry",
    *,
    delta: float = 1.2,
    override_bonding_capacities: dict[str, int] | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
    """Calculate the geometry adjacency matrix.

    Parameters
    ----------
    geo
        Geometry.
    delta
        Factor to scale covalent matrices for bond consideration.
        `bond_cutoff = delta * (r_covalent1 + r_covalent2)`
    override_bonding_capacities
        {"symbol": capacity} for overriding max coordination numbers.

    Returns
    -------
    np.ndarray
        Adjacency matrix.
    """
    dmat = distance_matrix(geo)
    radii = np.array(geo.covalent_radii)

    r_cov_matrix = delta * (radii[:, np.newaxis] + radii[np.newaxis, :])
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


def dihedral_angle(
    geo: "Geometry", keys: Sequence[int], *, degrees: bool = True
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


def inertia_tensor(geo: "Geometry") -> FloatArray:
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


def inertia_moments(geo: "Geometry") -> FloatArray:
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


def inertia_axes(geo: "Geometry") -> FloatArray:
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


def mass_weight_vector(geo: "Geometry") -> np.ndarray:
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


def rotational_analysis(geo: "Geometry") -> tuple[FloatArray, FloatArray]:
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
    geo: "Geometry", *, mass_weighted: bool = True
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
    geo: "Geometry", hess: FloatArray, *, trans: bool = False, rot: bool = False
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
    """Calculate the harmonic zero point vibrational energy of a geometry."""
    if freqs is None:
        freqs, _ = vibrational_analysis(geo, np.array(hess))

    zpe_wavenumbers = 0.5 * sum([f for f in freqs if f > 0.0])

    return zpe_wavenumbers * constants.WAVENUMBER_TO_HARTREE
