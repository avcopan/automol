"""Geometry transformations."""

from collections.abc import Collection
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from stereomolgraph import StereoCondensedReactionGraph

from .core import from_stereo_mol_graph, stereo_mol_graph

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


def transition(geo1: "Geometry", geo2: "Geometry") -> "Geometry":
    """Determine the transition geometry between two geometries.

    Parameters
    ----------
    geo1
        Initial geometry.
    geo2
        Final geometry.

    Returns
    -------
        Geometry.
    """
    if geo1.spin != geo2.spin:
        msg = f"Geometries must have the same spin: {geo1.spin} != {geo2.spin}"
        raise ValueError(msg)

    smg1 = stereo_mol_graph(geo1)
    smg2 = stereo_mol_graph(geo2)
    scrg = StereoCondensedReactionGraph.from_graphs(smg1, smg2)

    active_h = [a for a in scrg.active_atoms() if scrg.get_atom_type(a) == 1]
    for h in active_h:
        scrg.set_atom_attribute(h, "atom_type", 8)

    ts_smg = scrg.ts()
    ts_geo = from_stereo_mol_graph(ts_smg)
    ts_geo.spin = geo1.spin

    for h in active_h:
        ts_geo.symbols[h] = "H"

    return ts_geo
