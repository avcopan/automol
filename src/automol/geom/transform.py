"""Geometry transformations."""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .properties import center_of_mass, inertia_axes

if TYPE_CHECKING:
    from .core import Geometry


def translate(
    geo: "Geometry",
    arr: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> "Geometry":
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
    geo: "Geometry",
    normal: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> "Geometry":
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
    geo: "Geometry",
    rot: Rotation,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> "Geometry":
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


def rotation_to_inertia_axes(geo: "Geometry") -> Rotation:
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
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    # Move to center of mass
    geo = translate(geo, -center_of_mass(geo), in_place=True)
    # Rotate to inertia axes
    rot = rotation_to_inertia_axes(geo)
    return rotate(geo, rot, in_place=True)
