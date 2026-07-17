"""Inertia and rotational analysis of a geometry."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .properties import center_of_mass
from .transform import rotate, translate

if TYPE_CHECKING:
    from .core import Geometry


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


def eckart_frame(geo: "Geometry", *, in_place: bool = False) -> "Geometry":
    """Rotate geometry to align with inertia axes.

    Parameters
    ----------
    geo
        Geometry.
    in_place
        Whether to rotate in place or return a new geometry.

    Returns
    -------
        Geometry in an Eckart frame.

    Example
    -------
    >>> from automol import Geometry
    >>> geo = Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> eck = eckart_frame(geo)
    >>> bool(np.allclose(center_of_mass(eck), 0, atol=1e-10))
    True
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    # Move to center of mass
    geo = translate(geo, -center_of_mass(geo), in_place=True)
    # Rotate to inertia axes
    rot = rotation_to_inertia_axes(geo)
    return rotate(geo, rot, in_place=True)
