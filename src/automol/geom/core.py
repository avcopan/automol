"""Molecular geometry functions."""

from collections import Counter
from pathlib import Path
from typing import Self

import numpy as np
import pyparsing as pp
from ase import Atoms
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from pyparsing import pyparsing_common as ppc
from rdkit.Chem import Mol
from stereomolgraph import StereoMolGraph
from stereomolgraph.coords import Geometry as SMGeometry

from .. import rd
from ..utils import element
from ..utils.exc import XYZFormatError
from ..utils.types import CoordinatesField

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

    try:
        symbs, coords = zip(
            *[XYZ_LINE.parse_string(line).as_list() for line in lines], strict=True
        )
    except pp.ParseException as exc:
        msg = f"Failed to parse xyz line: {exc.line!r}"
        raise XYZFormatError(msg) from exc

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
    smg = stereo_mol_graph(geo)
    return smg.to_rdmol(charge=geo.charge)


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


def to_ase(geo: Geometry) -> Atoms:
    """Instantiate an ASE Atoms object from a Geometry."""
    return Atoms(
        symbols=geo.symbols,
        positions=geo.coordinates,
        info={"charge": geo.charge, "spin": geo.spin},
    )


def stereo_mol_graph(geo: Geometry) -> StereoMolGraph:
    """Instantiate a StereoMolGraph from a Geometry."""
    sm_geo = SMGeometry(atom_types=tuple(geo.symbols), coords=geo.coordinates)
    return StereoMolGraph.from_geometry(sm_geo)  # ty:ignore[invalid-argument-type]


def hill_formula(geo: Geometry) -> str:
    """Render the molecular formula in Hill order."""
    counts = Counter(s.capitalize() for s in geo.symbols)

    ordered = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    ordered.extend(sorted(counts.items(), key=lambda x: x[0]))

    return "".join(s if n == 1 else f"{s}{n}" for s, n in ordered)
