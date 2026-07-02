"""Molecular geometry functions."""

import hashlib
from pathlib import Path
from typing import Self

import numpy as np
import pyparsing as pp
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pyparsing import pyparsing_common as ppc
from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds

from .. import element, rd
from ..utils.exc import GeometryConversionError, XYZFormatError
from ..utils.types import CoordinatesField
from .canon import canonical_frame, canonical_sorting

CHAR = pp.Char(pp.alphas)
SYMBOL = pp.Combine(CHAR + pp.Opt(CHAR))
XYZ_LINE = SYMBOL + pp.Group(ppc.fnumber * 3) + pp.Suppress(... + pp.LineEnd())


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
    hash: str | None = None

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

    @model_validator(mode="after")
    def set_hash(self) -> Self:
        """Populate hash after model validation."""
        if not all(
            getattr(self, field, None) is not None
            for field in ("symbols", "coordinates", "charge", "spin")
        ):
            return self

        object.__setattr__(self, "hash", geometry_hash(self, decimals=4))
        return self

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

    def canonical_form(
        self: Self, *, delta: float = 1.4, decimals: int = 4, truncate: bool = True
    ) -> Self:
        """Return a canonical form of the geometry.

        Parameters
        ----------
        delta
            Factor to scale covalent matrices for bond consideration.
            `bond_cutoff = delta * (r_covalent1 + r_covalent2)`
        decimals
            Number of decimal places to consider in sort tie-breaking.
        truncate
            If True, truncate hypervalent bonds by highest covalent radius deviation.

        Returns
        -------
        Self
            Canonical form of the geometry.
        """
        # Resort indices by canonical ordering
        canon_idxs = canonical_sorting(
            self, delta=delta, decimals=decimals, truncate=truncate
        )
        temp_geo = Geometry(
            symbols=[self.symbols[i] for i in canon_idxs],
            coordinates=self.coordinates[canon_idxs],
            charge=self.charge,
            spin=self.spin,
        )

        # Re-orient the coordinates by canonical framing
        canon_geo = canonical_frame(temp_geo, decimals=decimals)

        return type(self)(
            symbols=canon_geo.symbols,
            coordinates=canon_geo.coordinates,
            charge=canon_geo.charge,
            spin=canon_geo.spin,
        )

    def xyz_block(self, *, comment: str | None = None) -> str:
        """Return Geometry as a formatted xyz block with optional comment."""
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
        """Write Geometry as a formatted xyz file with optional comment."""
        xyz_file(self, path=path, comment=comment)

    @classmethod
    def from_xyz_file(cls, path: str | Path, *, charge: int, spin: int) -> Self:
        """Instantiate Geometry from a formatted xyz file."""
        path = Path(path)
        return cls.from_xyz_block(path.read_text(), charge=charge, spin=spin)


def geometry_hash(geo: Geometry, decimals: int = 4) -> str:
    """Generate a deterministic geometry hash string.

    Parameters
    ----------
    decimals
        Number of decimal places to round the coordinates before hashing.

    Returns
    -------
        Geometry hash string.
    """
    # 1. Convert symbols and coordinates to integers
    numbers = geo.atomic_numbers
    icoords = np.rint(geo.coordinates * 10**decimals)

    # 2. Generate bytes representation of each field
    numbers_bytes = np.asarray(numbers, dtype=np.dtype("<i8")).tobytes("C")
    icoords_bytes = icoords.astype(np.dtype("<i8")).tobytes("C")
    charge_bytes = geo.charge.to_bytes(1, byteorder="little", signed=True)
    spin_bytes = geo.spin.to_bytes(1, byteorder="little", signed=True)

    # 3. Combine all bytes and generate hash
    geo_bytes = b"|".join([numbers_bytes, icoords_bytes, charge_bytes, spin_bytes])

    return hashlib.sha256(geo_bytes).hexdigest()


def xyz_block(geo: Geometry, *, comment: str | None = None) -> str:
    """Return Geometry as a formatted xyz block with optional comment."""
    lines = [str(geo.atom_count), comment or ""]
    for sym, (x, y, z) in zip(geo.symbols, geo.coordinates, strict=True):
        lines.append(f"{sym:<4} {x:12.8f} {y:12.8f} {z:12.8f}")

    return "\n".join(lines)


def from_xyz_block(xyz_block: str, *, charge: int, spin: int) -> Geometry:
    """Instantiate Geometry from a formatted xyz block."""
    lines = xyz_block.strip().splitlines()[2:]

    if not lines:
        msg = "The provided xyz block is empty."
        raise XYZFormatError(msg)

    symbs, coords = zip(
        *[XYZ_LINE.parse_string(line).as_list() for line in lines], strict=True
    )

    return Geometry(
        symbols=list(symbs), coordinates=np.array(coords), charge=charge, spin=spin
    )


def xyz_file(geo: Geometry, *, path: str | Path, comment: str | None = None) -> None:
    """Write a Geometry to a formatted xyz file with optional comment."""
    Path(path).write_text(xyz_block(geo, comment=comment))


def from_xyz_file(path: str | Path, *, charge: int, spin: int) -> Geometry:
    """Instantiate Geometry from a formatted xyz file."""
    return from_xyz_block(Path(path).read_text(), charge=charge, spin=spin)


def rdkit_mol(geo: Geometry) -> Mol:
    """Instantiate an rdkit Mol from a Geometry."""
    raw_mol = Chem.MolFromXYZBlock(xyzBlock=xyz_block(geo))
    conn_mol = Chem.Mol(raw_mol)

    # Determine connectivity (graph) only -- independent of charge/spin.
    rdDetermineBonds.DetermineConnectivity(conn_mol, useHueckel=True)

    # Try the true charge first; some radicals still resolve
    # if RDKit leaves deficiencies as implicit-Hs.
    last_err: Exception | None = None
    for trial_charge in (
        geo.charge,
        geo.charge - geo.spin,
        geo.charge + geo.spin,
    ):
        trial_mol = Chem.Mol(conn_mol)
        try:
            rdDetermineBonds.DetermineBondOrders(
                trial_mol, charge=trial_charge, allowChargedFragments=True
            )
        except ValueError as err:
            last_err = err
            continue

        n_placed = 0
        for a in trial_mol.GetAtoms():
            charge = a.GetFormalCharge()
            if charge != 0:
                a.SetNumRadicalElectrons(abs(charge))
                a.SetFormalCharge(0)
                n_placed += abs(charge)

        if n_placed == geo.spin:
            return trial_mol

    msg = f"Could not determine bond orders with {geo.charge = }, {geo.spin = }."
    raise GeometryConversionError(msg) from last_err


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
