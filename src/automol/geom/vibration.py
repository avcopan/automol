"""Vibrational normal-mode and frequency analysis of a geometry."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..utils import constants
from .inertia import rotational_analysis
from .properties import center_of_mass

if TYPE_CHECKING:
    from .core import Geometry


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
