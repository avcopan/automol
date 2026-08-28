"""Molecular geometry functions."""

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from rdkit.Chem import Mol
from scipy.spatial.transform import Rotation
from stereomolgraph import StereoCondensedReactionGraph, StereoMolGraph
from stereomolgraph.coords import Geometry as SMGeometry

from .. import rd
from ..utils import element
from ..utils.types import CoordinatesField
from .analysis import center_of_mass, rotation_to_inertia_axes
from .io import from_xyz_block, xyz_block, xyz_file


class Geometry(BaseModel):
    """
    Molecular geometry.

    Parameters
    ----------
    symbols
        Atomic symbols in order (e.g., ``["H", "O", "H"]``).
        The length of ``symbols`` must match the number of atoms.
    coordinates
        Cartesian coordinates of the atoms in Angstroms.
        Shape is ``(len(symbols), 3)`` and the ordering corresponds to ``symbols``.
    charge
        Total molecular charge.
    spin
        Number of unpaired electrons, i.e. two times the spin quantum number (``2S``).

    Example
    -------
    ```
    h2o = Geometry(
        symbols = ["H", "O", "H"],
        coordinates = [[0.0, 0.0, -0.74], [0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge = 0,
        spin = 0,
    )
    ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int
    spin: int

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates_shape(
        cls: Self, v: CoordinatesField, info: ValidationInfo
    ) -> CoordinatesField:
        """Validate shape of geometry coordinates."""
        symbols = info.data.get("symbols")
        if symbols is not None and v is not None and len(symbols) != v.shape[0]:
            msg = (
                f"Number of symbols ({len(symbols)}) does not match coordinates"
                f"{v.shape[0]}."
            )
            raise ValueError(msg)

        return v

    @property
    def atom_count(self) -> int:
        """Get number of atoms."""
        return len(self.symbols)

    @property
    def masses(self) -> list[float]:
        """Get isotopic masses."""
        return list(map(element.mass, self.symbols))

    @property
    def atomic_numbers(self) -> list[int]:
        """Get atomic numbers."""
        return list(map(element.number, self.symbols))

    @property
    def covalent_radii(self) -> list[float]:
        """Get Pyykko covalent radii in A."""
        return list(map(element.covalent_radius, self.symbols))

    @property
    def valences(self) -> list[int]:
        """Get numbers of valence electrons."""
        return list(map(element.valence, self.symbols))

    def _repr_html_(self) -> str | None:
        """Render geometry inline in Jupyter."""
        from . import io  # noqa: PLC0415  (avoids circular import)

        return io.view(self, label=True)._repr_html_()

    def __repr__(self) -> str:
        """Render Geometry as an xyz block instead of dumping raw fields."""
        return self.xyz_block()

    __str__ = __repr__

    def xyz_block(self, *, comment: str | None = None) -> str:
        """Return Geometry as a formatted xyz block.

        Defaults to a comment reporting the charge and spin, e.g. "Geometry(q=0, s=0)".
        """
        return xyz_block(self, comment=comment)

    @classmethod
    def from_xyz_block(cls, xyz_block: str, *, charge: int, spin: int) -> Self:
        """Instantiate Geometry from a formatted xyz block."""
        base_geo = from_xyz_block(xyz_block, charge=charge, spin=spin)
        return cls(
            symbols=base_geo.symbols,
            coordinates=base_geo.coordinates,
            charge=base_geo.charge,
            spin=base_geo.spin,
        )

    def xyz_file(self, *, path: str | Path, comment: str | None = None) -> None:
        """Write Geometry as a formatted xyz file.

        Defaults to a comment reporting the charge and spin, e.g. "Geometry(q=0, s=0)".
        """
        xyz_file(self, path=path, comment=comment)

    @classmethod
    def from_xyz_file(cls, path: str | Path, *, charge: int, spin: int) -> Self:
        """Instantiate Geometry from a formatted xyz file."""
        path = Path(path)
        return cls.from_xyz_block(path.read_text(), charge=charge, spin=spin)

    def relabel_atoms(self, indices: list[int] | tuple[int, ...]) -> Self:
        """Reorder atoms according to the provided indices.

        Parameters
        ----------
        indices
            Sequence of indices specifying the new atom order.
            E.g., [2, 0, 1] moves atom 2 to position 0, atom 0 to position 1, etc.

        Returns
        -------
        Reordered Geometry.
        """
        indices_array = np.array(indices)
        new_symbols = [self.symbols[i] for i in indices_array]
        new_coordinates = self.coordinates[indices_array]

        return self.__class__(
            symbols=new_symbols,
            coordinates=new_coordinates,
            charge=self.charge,
            spin=self.spin,
        )


def rdkit_mol(geo: Geometry) -> Mol:
    """Instantiate an rdkit Mol from a Geometry."""
    smg = stereo_mol_graph(geo)
    mol = smg.to_rdmol(charge=geo.charge)
    return rd.mol.set_coordinates(mol, geo.coordinates, in_place=True)


def from_rdkit_mol(mol: Mol) -> Geometry:
    """Instantiate a Geometry from an rdkit molecule."""
    if not rd.mol.has_coordinates(mol):
        mol = rd.mol.add_coordinates(mol)

    return Geometry(
        symbols=rd.mol.symbols(mol),
        coordinates=rd.mol.coordinates(mol),
        charge=rd.mol.charge(mol),
        spin=rd.mol.spin(mol),
    )


def stereo_mol_graph(geo: Geometry) -> StereoMolGraph:
    """Instantiate a StereoMolGraph from a Geometry."""
    sm_geo = SMGeometry(atom_types=tuple(geo.symbols), coords=geo.coordinates)
    return StereoMolGraph.from_geometry(sm_geo)  # ty:ignore[invalid-argument-type]


def from_stereo_mol_graph(smg: StereoMolGraph, *, charge: int = 0) -> Geometry:
    """Instantiate a Geometry from a StereoMolGraph."""
    mol = smg.to_rdmol(charge=charge)
    return from_rdkit_mol(mol)


def set_bond(
    geo: Geometry,
    *,
    idxs: Sequence[int],
    val: float,
    max_change: float = 0.25,
    in_place: bool = False,
) -> Geometry:
    """
    Set bond distance between two atoms.

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


# Rigid-body transformations
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


def transition(geo1: Geometry, geo2: Geometry) -> Geometry:
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
