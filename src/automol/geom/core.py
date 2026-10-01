"""Molecular geometry functions."""

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from rdkit.Chem import Mol
from scipy.spatial.transform import Rotation

from .. import rd
from ..utils import element
from ..utils.exc import ElementNotFoundError
from ..utils.types import CoordinatesField
from .analysis import adjacency_matrix
from .io import from_xyz_block, view, xyz_block, xyz_file


class Geometry(BaseModel):
    """
    Molecular geometry.

    Parameters
    ----------
    symbols
        Atomic symbols in order (e.g., ``["H", "O", "H"]``).
        The length of ``symbols`` must match the number of atoms. Symbols are
        validated against the periodic table and canonicalized (``"cl"`` -> ``"Cl"``).
    coordinates
        Cartesian coordinates of the atoms in Angstroms.
        Shape is ``(len(symbols), 3)`` and the ordering corresponds to ``symbols``.
    charge
        Total molecular charge.
    spin
        Number of unpaired electrons, i.e. two times the spin quantum number (``2S``).
        Must be non-negative and have the same parity as the electron count.

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

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate and canonicalize atomic symbols."""
        try:
            return [element.symbol(s) for s in v]
        except ElementNotFoundError as err:
            raise ValueError(str(err)) from err

    # Cross-field checks are field validators (not a model validator) so that a
    # failed assignment under `validate_assignment` leaves the model unchanged.
    @field_validator("symbols", "coordinates", "charge", "spin")
    @classmethod
    def validate_consistency(cls, v: Any, info: ValidationInfo) -> Any:  # noqa: ANN401
        """Validate consistency of symbols, coordinates, charge, and spin."""
        data = {**info.data, str(info.field_name): v}
        symbols = data.get("symbols")
        coordinates = data.get("coordinates")
        if symbols is not None and coordinates is not None:
            _check_atom_count(symbols, coordinates)
        if symbols is not None and "charge" in data and "spin" in data:
            _check_spin(symbols, charge=data["charge"], spin=data["spin"])
        return v

    def __eq__(self, other: object) -> bool:
        """Compare geometries field by field (coordinates compared exactly)."""
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _values_equal(getattr(self, name), getattr(other, name))
            for name in type(self).model_fields
        )

    # Mutable (validate_assignment) model with array data, so not hashable
    __hash__ = None

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
        return view(self, label=True)._repr_html_()

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

    def relabel_atoms(self, indices: Sequence[int]) -> Self:
        """Reorder atoms according to the provided indices.

        Parameters
        ----------
        indices
            Permutation of ``range(atom_count)`` specifying the new atom order.
            E.g., [2, 0, 1] moves atom 2 to position 0, atom 0 to position 1, etc.

        Returns
        -------
        Reordered Geometry (other fields, including those of subclasses, are kept).

        Raises
        ------
        ValueError
            If ``indices`` is not a permutation of ``range(atom_count)``.
        """
        indices = [int(i) for i in indices]
        if sorted(indices) != list(range(self.atom_count)):
            msg = f"{indices} is not a permutation of range({self.atom_count})."
            raise ValueError(msg)

        return self.model_copy(
            deep=True,
            update={
                "symbols": [self.symbols[i] for i in indices],
                "coordinates": self.coordinates[indices],
            },
        )


def _check_atom_count(symbols: Sequence[str], coordinates: np.ndarray) -> None:
    """Check that there is one row of coordinates per symbol."""
    if len(symbols) != coordinates.shape[0]:
        msg = (
            f"Number of symbols ({len(symbols)}) does not match number of "
            f"coordinates ({coordinates.shape[0]})."
        )
        raise ValueError(msg)


def _check_spin(symbols: Sequence[str], *, charge: int, spin: int) -> None:
    """Check that spin is consistent with the electron count."""
    nelec = sum(map(element.number, symbols)) - charge
    if spin < 0 or nelec < 0 or spin > nelec or (nelec - spin) % 2:
        msg = f"Spin {spin} is inconsistent with {nelec} electrons (charge {charge})."
        raise ValueError(msg)


def _values_equal(val1: Any, val2: Any) -> bool:  # noqa: ANN401
    """Compare two field values, handling numpy arrays."""
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        return bool(np.array_equal(val1, val2))
    return bool(val1 == val2)


def rdkit_mol(geo: Geometry) -> Mol:
    """Instantiate an rdkit Mol from a Geometry.

    Connectivity is determined from covalent radii. Bond orders, formal charges, and
    radicals are then perceived to match the geometry's charge and spin (see
    `automol.rd.mol.from_connectivity`), and stereochemistry is assigned from the 3D
    coordinates.

    Raises
    ------
    NotImplementedError
        If the geometry contains a metal.
    GeometryConversionError
        If no Lewis structure is consistent with the connectivity, charge, and spin.
    """
    amat = adjacency_matrix(geo, enforce_valence=True)
    bonds = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(amat)), strict=True)]
    mol = rd.mol.from_connectivity(
        geo.symbols,
        bonds,
        charge=geo.charge,
        spin=geo.spin,
        coords=geo.coordinates,
    )
    return rd.mol.assign_stereochemistry(mol, in_place=True)


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
    keys
        Atoms to translate. If None, translate all atoms.
    in_place
        Whether to translate in place or return a new geometry.

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
        Normal vector of the reflection plane (which passes through the origin).
    keys
        Atoms to reflect. If None, reflect all atoms.
    in_place
        Whether to reflect in place or return a new geometry.

    Returns
    -------
        Geometry.

    Raises
    ------
    ValueError
        If ``normal`` is a zero vector.
    """
    normal = np.asarray(normal, dtype=float)
    norm2 = np.dot(normal, normal)
    if not norm2 > 0:
        msg = f"Reflection plane normal must be a non-zero vector, got {normal}."
        raise ValueError(msg)

    geo = geo if in_place else geo.model_copy(deep=True)
    proj = np.outer(normal, normal) / norm2
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
