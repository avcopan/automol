"""The `Geometry` model and the algorithms that operate on it."""

import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import py3Dmol
import pyparsing as pp
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator
from pyparsing import pyparsing_common as ppc
from rdkit.Chem import Mol
from scipy import spatial
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from . import rd
from .utils import element
from .utils.exc import ElementNotFoundError, XYZFormatError
from .utils.types import CoordinatesField, FloatArray

if TYPE_CHECKING:
    import xyzrender

__all__ = [
    "Geometry",
    "View",
    "adjacency_matrix",
    "alignment_rotation",
    "angle",
    "angles",
    "axis_alignment_rotation",
    "bonds",
    "center",
    "center_of_mass",
    "dihedral",
    "dihedrals",
    "distance",
    "distance_keys",
    "distance_matrix",
    "from_rdkit_mol",
    "from_xyz_block",
    "from_xyz_file",
    "from_xyz_trajectory_block",
    "from_xyz_trajectory_file",
    "inertia_tensor",
    "inertial_frame_rotation",
    "join",
    "principal_axes",
    "rdkit_mol",
    "reflect",
    "remove_atoms",
    "render_gif",
    "render_svg",
    "rotate",
    "select_atoms",
    "translate",
    "twist",
    "twist_angle",
    "view",
    "xyz_block",
    "xyz_file",
    "xyz_trajectory_block",
    "xyz_trajectory_comments",
    "xyz_trajectory_file",
]


class Geometry(BaseModel):
    """Molecular geometry.

    Attributes:
        symbols: Atomic symbols, e.g. ``["H", "O", "H"]``. Validated against the
            periodic table and canonicalized (``"cl"`` -> ``"Cl"``).
        coordinates: Cartesian coordinates in Angstroms, shape ``(N, 3)`` and
            ordered like `symbols`. A `pint.Quantity` with length units is
            converted to Angstroms; plain values are assumed to be Angstroms.
        charge: Total molecular charge.
        spin: Number of unpaired electrons (``2S``). Must be non-negative and
            have the same parity as the electron count.

    Example:
        >>> geo = Geometry(
        ...     symbols=["H", "O", "H"],
        ...     coordinates=[[0.0, 0.0, -0.74], [0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> geo.atom_count
        3
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int
    spin: int

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate and canonicalize atomic symbols.

        Args:
            v: Atomic symbols.

        Returns:
            Canonicalized atomic symbols.

        Raises:
            ValueError: If any symbol is not a known element.
        """
        try:
            return [element.symbol(s) for s in v]
        except ElementNotFoundError as err:
            raise ValueError(str(err)) from err

    # A field validator (not a model validator), so that a failed assignment
    # under `validate_assignment` leaves the model unchanged
    @field_validator("symbols", "coordinates", "charge", "spin")
    @classmethod
    def validate_consistency(cls, v: Any, info: ValidationInfo) -> Any:  # noqa: ANN401
        """Check that symbols, coordinates, charge, and spin are consistent.

        Args:
            v: Candidate field value.
            info: Validation context.

        Returns:
            The unchanged value.
        """
        data = {**info.data, str(info.field_name): v}
        symbols = data.get("symbols")
        coordinates = data.get("coordinates")
        if symbols is not None and coordinates is not None:
            _check_atom_count(symbols, coordinates)
        if symbols is not None and "charge" in data and "spin" in data:
            _check_spin(symbols, charge=data["charge"], spin=data["spin"])
        return v

    def __eq__(self, other: object) -> bool:
        """Compare geometries field by field, including exact coordinates."""
        if type(other) is not type(self):
            return NotImplemented
        return all(
            _values_equal(getattr(self, name), getattr(other, name))
            for name in type(self).model_fields
        )

    # Mutable model with array data, so not hashable
    __hash__ = None

    def __repr__(self) -> str:
        """Render the geometry as an xyz block."""
        return self.xyz_block()

    __str__ = __repr__

    def _repr_html_(self) -> str | None:
        """Render the geometry inline in Jupyter."""
        return view(self, label=True)._repr_html_()

    @property
    def atom_count(self) -> int:
        """Number of atoms."""
        return len(self.symbols)

    @property
    def masses(self) -> list[float]:
        """Atomic masses."""
        return list(map(element.mass, self.symbols))

    @property
    def atomic_numbers(self) -> list[int]:
        """Atomic numbers."""
        return list(map(element.number, self.symbols))

    @property
    def covalent_radii(self) -> list[float]:
        """Pyykko covalent radii, in Angstroms."""
        return list(map(element.covalent_radius, self.symbols))

    @property
    def valences(self) -> list[int]:
        """Numbers of valence electrons."""
        return list(map(element.valence, self.symbols))

    def xyz_block(self, *, comment: str | None = None) -> str:
        """Format the geometry as an xyz block (see `xyz_block`)."""
        return xyz_block(self, comment=comment)

    def xyz_file(self, *, path: str | Path, comment: str | None = None) -> None:
        """Write the geometry to an xyz file (see `xyz_file`)."""
        xyz_file(self, path=path, comment=comment)

    @classmethod
    def from_xyz_block(cls, xyz_block: str, *, charge: int, spin: int) -> Self:
        """Parse a geometry from an xyz block (see `from_xyz_block`)."""
        symbols, coordinates = _parse_xyz_block(xyz_block)
        return cls(symbols=symbols, coordinates=coordinates, charge=charge, spin=spin)

    @classmethod
    def from_xyz_file(cls, path: str | Path, *, charge: int, spin: int) -> Self:
        """Read a geometry from an xyz file (see `from_xyz_file`)."""
        return cls.from_xyz_block(Path(path).read_text(), charge=charge, spin=spin)

    def relabel_atoms(self, indices: Sequence[int]) -> Self:
        """Reorder the atoms.

        Args:
            indices: Permutation of ``range(atom_count)`` giving the new order,
                e.g. ``[2, 0, 1]`` moves atom 2 to position 0, atom 0 to
                position 1, and atom 1 to position 2.

        Returns:
            A reordered copy. Other fields, including those of subclasses, are
            kept.

        Raises:
            ValueError: If `indices` is not a permutation of ``range(atom_count)``.
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
    """Raise `ValueError` unless there is one row of coordinates per symbol."""
    if len(symbols) != coordinates.shape[0]:
        msg = (
            f"Number of symbols ({len(symbols)}) does not match number of "
            f"coordinates ({coordinates.shape[0]})."
        )
        raise ValueError(msg)


def _check_spin(symbols: Sequence[str], *, charge: int, spin: int) -> None:
    """Raise `ValueError` unless `spin` is consistent with the electron count."""
    nelec = sum(map(element.number, symbols)) - charge
    if spin < 0 or nelec < 0 or spin > nelec or (nelec - spin) % 2:
        msg = f"Spin {spin} is inconsistent with {nelec} electrons (charge {charge})."
        raise ValueError(msg)


def _values_equal(val1: Any, val2: Any) -> bool:  # noqa: ANN401
    """Compare two field values, handling numpy arrays."""
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        return bool(np.array_equal(val1, val2))
    return bool(val1 == val2)


# XYZ I/O
# Each token must be followed by whitespace or the end of the line, so that
# e.g. "H1 0 0 0" is rejected rather than parsed as H at (1, 0, 0)
_END = r"(?=\s|$)"
_SYMBOL = pp.Regex(rf"(?:[A-Za-z]{{1,2}}|\d{{1,3}}){_END}")
_FLOAT = pp.Regex(rf"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?{_END}")
_FLOAT.set_parse_action(ppc.convert_to_float)
_XYZ_LINE = _SYMBOL + pp.Group(_FLOAT * 3) + pp.Suppress(... + pp.LineEnd())


def xyz_block(geo: Geometry, *, comment: str | None = None) -> str:
    """Format a geometry as an xyz block.

    Args:
        geo: Geometry.
        comment: Comment line. Defaults to the charge and spin, e.g.
            ``"Geometry(q=0, s=0)"``.

    Returns:
        The xyz block.

    Raises:
        ValueError: If `comment` contains a newline.
    """
    if comment is None:
        comment = f"Geometry(q={geo.charge}, s={geo.spin})"
    if "\n" in comment or "\r" in comment:
        msg = f"The xyz comment must be a single line, got {comment!r}."
        raise ValueError(msg)
    lines = [str(geo.atom_count), comment]
    for sym, (x, y, z) in zip(geo.symbols, geo.coordinates.tolist(), strict=True):
        lines.append(f"{sym:<4} {x:12.8f} {y:12.8f} {z:12.8f}")

    return "\n".join(lines)


def from_xyz_block(xyz_block: str, *, charge: int, spin: int) -> Geometry:
    """Parse a geometry from a single-frame xyz block.

    Atoms may be given by symbol (case-insensitive) or atomic number.

    Args:
        xyz_block: The xyz block.
        charge: Total molecular charge.
        spin: Number of unpaired electrons.

    Returns:
        The parsed geometry.

    Raises:
        XYZFormatError: If the block is empty, the atom count line is missing or
            does not match the number of atom lines, or an atom line cannot be
            parsed.
    """
    return Geometry.from_xyz_block(xyz_block, charge=charge, spin=spin)


def xyz_file(geo: Geometry, *, path: str | Path, comment: str | None = None) -> None:
    """Write a geometry to an xyz file.

    Args:
        geo: Geometry.
        path: Output path.
        comment: Comment line. Defaults to the charge and spin, e.g.
            ``"Geometry(q=0, s=0)"``.
    """
    Path(path).write_text(xyz_block(geo, comment=comment) + "\n")


def from_xyz_file(path: str | Path, *, charge: int, spin: int) -> Geometry:
    """Read a geometry from an xyz file (see `from_xyz_block`).

    Args:
        path: Path to the xyz file.
        charge: Total molecular charge.
        spin: Number of unpaired electrons.

    Returns:
        The parsed geometry.
    """
    return Geometry.from_xyz_file(path, charge=charge, spin=spin)


def xyz_trajectory_block(
    geos: Sequence[Geometry], *, comments: Sequence[str] | None = None
) -> str:
    """Format geometries as a multi-frame xyz block, one frame per geometry.

    Args:
        geos: Geometries.
        comments: Comment line for each frame. Defaults to the charge and spin
            (see `xyz_block`).

    Returns:
        The xyz block.

    Raises:
        ValueError: If `comments` doesn't have one entry per geometry.
    """
    frame_comments: Sequence[str | None] = (
        [None] * len(geos) if comments is None else comments
    )
    if len(frame_comments) != len(geos):
        msg = f"Got {len(frame_comments)} comments for {len(geos)} geometries."
        raise ValueError(msg)
    return "\n".join(
        xyz_block(geo, comment=comment)
        for geo, comment in zip(geos, frame_comments, strict=True)
    )


def from_xyz_trajectory_block(
    xyz_block: str, *, charge: int, spin: int
) -> list[Geometry]:
    """Parse geometries from a multi-frame xyz block, e.g. a list of conformers.

    Each frame is an atom count line, a comment line, and the atom lines (see
    `from_xyz_block`). A single-frame block gives a list of one geometry.

    Args:
        xyz_block: The xyz block.
        charge: Total molecular charge, for every frame.
        spin: Number of unpaired electrons, for every frame.

    Returns:
        The parsed geometries, in order.

    Raises:
        XYZFormatError: If the block is empty, or a frame is malformed.
    """
    return [
        from_xyz_block(frame, charge=charge, spin=spin)
        for frame in _split_xyz_frames(xyz_block)
    ]


def xyz_trajectory_comments(xyz_block: str) -> list[str]:
    """Get the comment line of each frame of a multi-frame xyz block.

    Programs often store a per-frame value there, such as CREST's energies.

    Args:
        xyz_block: The xyz block.

    Returns:
        The comment lines, stripped of surrounding whitespace, in order.

    Raises:
        XYZFormatError: If the block is empty, or a frame is malformed.
    """
    return [frame.splitlines()[1].strip() for frame in _split_xyz_frames(xyz_block)]


def xyz_trajectory_file(
    geos: Sequence[Geometry],
    *,
    path: str | Path,
    comments: Sequence[str] | None = None,
) -> None:
    """Write geometries to a multi-frame xyz file (see `xyz_trajectory_block`).

    Args:
        geos: Geometries.
        path: Output path.
        comments: Comment line for each frame. Defaults to the charge and spin.
    """
    Path(path).write_text(xyz_trajectory_block(geos, comments=comments) + "\n")


def from_xyz_trajectory_file(
    path: str | Path, *, charge: int, spin: int
) -> list[Geometry]:
    """Read geometries from a multi-frame xyz file (see `from_xyz_trajectory_block`).

    Args:
        path: Path to the xyz file.
        charge: Total molecular charge, for every frame.
        spin: Number of unpaired electrons, for every frame.

    Returns:
        The parsed geometries, in order.
    """
    return from_xyz_trajectory_block(Path(path).read_text(), charge=charge, spin=spin)


def _split_xyz_frames(xyz_block: str) -> list[str]:
    """Split a multi-frame xyz block into single-frame blocks."""
    lines = xyz_block.strip().splitlines()
    if not lines:
        msg = "The provided xyz block is empty."
        raise XYZFormatError(msg)

    frames = []
    start = 0
    while start < len(lines):
        try:
            natms = int(lines[start])
        except ValueError as exc:
            msg = f"Expected an atom count on line {start + 1}, got {lines[start]!r}."
            raise XYZFormatError(msg) from exc
        stop = start + natms + 2
        if stop > len(lines):
            msg = (
                f"The frame starting on line {start + 1} has {natms} atoms, but "
                f"only {len(lines) - start - 2} atom lines follow."
            )
            raise XYZFormatError(msg)
        frames.append("\n".join(lines[start:stop]))
        start = stop
    return frames


def _parse_xyz_block(xyz_block: str) -> tuple[list[str], FloatArray]:
    """Parse symbols and coordinates from an xyz block (see `from_xyz_block`)."""
    lines = xyz_block.strip().splitlines()
    if not lines:
        msg = "The provided xyz block is empty."
        raise XYZFormatError(msg)

    try:
        natms = int(lines[0])
    except ValueError as exc:
        msg = f"Expected an atom count on the first line, got {lines[0]!r}."
        raise XYZFormatError(msg) from exc

    atom_lines = lines[2:]
    if not atom_lines:
        msg = "The provided xyz block contains no atoms."
        raise XYZFormatError(msg)

    if len(atom_lines) != natms:
        msg = (
            f"Atom count ({natms}) does not match the number of atom lines "
            f"({len(atom_lines)}). Multi-frame xyz blocks are not supported."
        )
        raise XYZFormatError(msg)

    try:
        symbs, coords = zip(
            *[_XYZ_LINE.parse_string(line).as_list() for line in atom_lines],
            strict=True,
        )
    except pp.ParseException as exc:
        msg = f"Failed to parse xyz line: {exc.line!r}"
        raise XYZFormatError(msg) from exc

    return [_parse_symbol(s) for s in symbs], np.array(coords, dtype=np.float64)


def _parse_symbol(token: str) -> str:
    """Convert an xyz atom token (symbol or atomic number) to a symbol."""
    try:
        return element.symbol(int(token) if token.isdigit() else token)
    except ElementNotFoundError as exc:
        msg = f"Unknown element in xyz block: {token!r}"
        raise XYZFormatError(msg) from exc


# RDKit conversion
def rdkit_mol(geo: Geometry) -> Mol:
    """Convert a geometry to an RDKit molecule.

    Connectivity comes from `adjacency_matrix` (with valence caps). Bond orders,
    formal charges, and radicals are perceived to match the geometry's charge
    and spin (see `automol.rd.mol.from_connectivity`), and stereochemistry is
    assigned from the 3D coordinates.

    Args:
        geo: Geometry.

    Returns:
        RDKit molecule with perceived bonding and stereochemistry.

    Raises:
        NotImplementedError: If the geometry contains a metal.
        GeometryConversionError: If no Lewis structure is consistent with the
            connectivity, charge, and spin.
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
    """Convert an RDKit molecule to a geometry, embedding coordinates if missing.

    Args:
        mol: RDKit molecule.

    Returns:
        The geometry.
    """
    if not rd.mol.has_coordinates(mol):
        mol = rd.mol.add_coordinates(mol)

    return Geometry(
        symbols=rd.mol.symbols(mol),
        coordinates=rd.mol.coordinates(mol),
        charge=rd.mol.charge(mol),
        spin=rd.mol.spin(mol),
    )


# Selecting and joining atoms
def select_atoms(
    geo: Geometry,
    keys: Sequence[int],
    *,
    charge: int | None = None,
    spin: int | None = None,
) -> Geometry:
    """Select a subset of atoms.

    Args:
        geo: Geometry.
        keys: Indices of the atoms to keep, in the order to keep them.
        charge: Charge of the selection. Defaults to the charge of `geo`.
        spin: Spin of the selection. Defaults to the spin of `geo` if that is
            consistent with the electron count, and otherwise to the lowest
            consistent spin (e.g. 1 for a methyl radical selected from methane).

    Returns:
        The selected atoms.

    Example:
        >>> geo = Geometry(
        ...     symbols=["C", "H", "H", "H", "H"],
        ...     coordinates=np.zeros((5, 3)),
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> methyl = select_atoms(geo, [0, 1, 2, 3])
        >>> methyl.symbols, methyl.spin
        (['C', 'H', 'H', 'H'], 1)
    """
    keys = [int(k) for k in keys]
    symbols = [geo.symbols[k] for k in keys]
    charge = geo.charge if charge is None else charge
    if spin is None:
        nelec = sum(map(element.number, symbols)) - charge
        spin = geo.spin if (nelec - geo.spin) % 2 == 0 else nelec % 2
    return Geometry(
        symbols=symbols,
        coordinates=geo.coordinates[keys],
        charge=charge,
        spin=spin,
    )


def remove_atoms(
    geo: Geometry,
    keys: Collection[int],
    *,
    charge: int | None = None,
    spin: int | None = None,
) -> Geometry:
    """Remove a subset of atoms.

    Args:
        geo: Geometry.
        keys: Indices of the atoms to remove.
        charge: Charge of the remaining atoms (see `select_atoms`).
        spin: Spin of the remaining atoms (see `select_atoms`).

    Returns:
        The remaining atoms, in their original order.
    """
    remove = set(map(int, keys))
    keep = [k for k in range(geo.atom_count) if k not in remove]
    return select_atoms(geo, keep, charge=charge, spin=spin)


def join(geos: Sequence[Geometry]) -> Geometry:
    """Join geometries into one.

    Args:
        geos: Geometries, in order.

    Returns:
        A geometry with the atoms of each geometry in turn, and the summed
        charges and spins.
    """
    return Geometry(
        symbols=[s for geo in geos for s in geo.symbols],
        coordinates=np.vstack([geo.coordinates for geo in geos]).reshape(-1, 3),
        charge=sum(geo.charge for geo in geos),
        spin=sum(geo.spin for geo in geos),
    )


# Properties
_FLOOD_FILL_STEP = 0.05


def center_of_mass(geo: Geometry) -> FloatArray:
    """Calculate the center of mass.

    Args:
        geo: Geometry.

    Returns:
        Center of mass coordinates (the origin for an empty geometry).
    """
    masses = np.asarray(geo.masses)
    if not masses.size:
        return np.zeros(3)
    return masses @ geo.coordinates / np.sum(masses)


def distance_matrix(geo: Geometry) -> np.ndarray:
    """Calculate the interatomic distance matrix.

    Args:
        geo: Geometry.

    Returns:
        Distance matrix, shape ``(N, N)``.
    """
    return spatial.distance.cdist(geo.coordinates, geo.coordinates)


def distance_keys(geo: Geometry) -> np.ndarray:
    """Build a sorted distance descriptor.

    Each unique atom pair gives a row ``[z_low, z_high, distance]``. Rows are
    sorted by atomic numbers, then by distance.

    Args:
        geo: Geometry.

    Returns:
        Descriptor array, shape ``(npairs, 3)``.
    """
    z = np.asarray(geo.atomic_numbers)
    dmat = distance_matrix(geo)
    iu, ju = np.triu_indices(geo.atom_count, k=1)

    pairs = np.sort(np.stack([z[iu], z[ju]], axis=1), axis=1)
    key = np.column_stack([pairs.astype(float), dmat[iu, ju]])

    return _sort_rows(key)


def adjacency_matrix(
    geo: Geometry,
    *,
    sigma: float = 1.3,
    flood_fill: bool = False,
    enforce_valence: bool = False,
) -> np.ndarray:
    """Determine connectivity from covalent radii.

    Two atoms are bonded if their distance is less than `sigma` times the sum of
    their covalent radii.

    Args:
        geo: Geometry.
        sigma: Scaling factor for the sum of covalent radii.
        flood_fill: If `True`, increase `sigma` in steps of 0.05 until the
            molecule is a single fragment, or until no more bonds can be added.
        enforce_valence: If `True`, cap the bonds to each atom at
            `automol.rd.mol.max_bond_count`. Bonds are accepted shortest first,
            relative to the sum of covalent radii, so the result does not depend
            on atom order.

    Returns:
        Symmetric 0/1 adjacency matrix, shape ``(N, N)``.
    """
    natms = geo.atom_count
    radii = np.asarray(geo.covalent_radii)
    iu, ju = np.triu_indices(natms, k=1)
    ratios = distance_matrix(geo)[iu, ju] / (radii[iu] + radii[ju])
    order = np.argsort(ratios, kind="stable")
    iu, ju, ratios = iu[order], ju[order], ratios[order]

    max_bonds = (
        [rd.mol.max_bond_count(s) for s in geo.symbols] if enforce_valence else None
    )

    while True:
        amat = np.zeros((natms, natms), dtype=int)
        ncand = int(np.searchsorted(ratios, sigma, side="left"))
        if max_bonds is None:
            amat[iu[:ncand], ju[:ncand]] = amat[ju[:ncand], iu[:ncand]] = 1
        else:
            counts = np.zeros(natms, dtype=int)
            for i, j in zip(iu[:ncand], ju[:ncand], strict=True):
                if counts[i] < max_bonds[i] and counts[j] < max_bonds[j]:
                    amat[i, j] = amat[j, i] = 1
                    counts[[i, j]] += 1

        if not flood_fill or ncand == len(ratios):
            break

        n_components, _ = connected_components(amat, directed=False)
        if n_components <= 1:
            break
        sigma += _FLOOD_FILL_STEP

    return amat


def inertia_tensor(geo: Geometry) -> np.ndarray:
    """Calculate the inertia tensor about the center of mass.

    Args:
        geo: Geometry.

    Returns:
        Inertia tensor, shape ``(3, 3)`` (zero for an empty geometry).

    Example:
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
    masses = np.asarray(geo.masses)
    coords = geo.coordinates - center_of_mass(geo)
    mr2 = np.einsum("i,ij,ij->", masses, coords, coords)
    return mr2 * np.eye(3) - np.einsum("i,ij,ik->jk", masses, coords, coords)


def principal_axes(geo: Geometry) -> tuple[FloatArray, FloatArray]:
    """Calculate the principal moments and axes of inertia.

    Args:
        geo: Geometry.

    Returns:
        The principal moments, in ascending order, and the principal axes, as
        the columns of a ``(3, 3)`` rotation matrix (a right-handed frame).
    """
    moments, axes = np.linalg.eigh(inertia_tensor(geo))
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return moments, axes


def inertial_frame_rotation(geo: Geometry) -> Rotation:
    """Find the rotation into the principal axis frame.

    After ``rotate(center(geo), rot)``, the principal axes with the smallest to
    largest moments lie along x, y, and z, so the inertia tensor is diagonal.

    Args:
        geo: Geometry.

    Returns:
        The rotation.
    """
    _, axes = principal_axes(geo)
    return Rotation.from_matrix(axes.T)


# Measurements
def distance(geo: Geometry, keys: Sequence[int]) -> float:
    """Measure the distance between two atoms.

    Args:
        geo: Geometry.
        keys: Indices of the two atoms.

    Returns:
        The distance, in Angstroms.
    """
    r1, r2 = _key_coordinates(geo, keys, 2)
    return float(np.linalg.norm(r2 - r1))


def angle(geo: Geometry, keys: Sequence[int], *, degrees: bool = False) -> float:
    """Measure the angle between three atoms.

    Args:
        geo: Geometry.
        keys: Indices of the three atoms, with the vertex in the middle.
        degrees: If `True`, return degrees instead of radians.

    Returns:
        The angle, between 0 and pi.

    Example:
        >>> geo = Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> angle(geo, [1, 0, 2], degrees=True)
        90.0
    """
    r1, r2, r3 = _key_coordinates(geo, keys, 3)
    v1, v2 = r1 - r2, r3 - r2
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ang = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(np.degrees(ang)) if degrees else ang


def dihedral(geo: Geometry, keys: Sequence[int], *, degrees: bool = False) -> float:
    """Measure the dihedral angle between four atoms.

    Looking down the 2-3 bond, the angle is positive when atom 4 is rotated
    clockwise from atom 1.

    Args:
        geo: Geometry.
        keys: Indices of the four atoms.
        degrees: If `True`, return degrees instead of radians.

    Returns:
        The dihedral angle, between -pi and pi.

    Example:
        >>> geo = Geometry(
        ...     symbols=["H", "O", "O", "H"],
        ...     coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> dihedral(geo, [0, 1, 2, 3], degrees=True)
        90.0
    """
    r1, r2, r3, r4 = _key_coordinates(geo, keys, 4)
    b1, b2, b3 = r2 - r1, r3 - r2, r4 - r3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
    ang = float(np.arctan2(y, x))
    return float(np.degrees(ang)) if degrees else ang


# Internal coordinates
def bonds(geo: Geometry, amat: ArrayLike) -> np.ndarray:
    """List bond distances.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        Sorted rows ``[z1, z2, distance]`` with ``z1 <= z2``, shape
        ``(nbonds, 3)``.

    Example:
        >>> geo = Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> b = bonds(geo, adjacency_matrix(geo))
        >>> b.shape
        (2, 3)
        >>> [tuple(float(x) for x in row) for row in b]
        [(1.0, 8.0, 1.0), (1.0, 8.0, 1.0)]
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    i, j = np.nonzero(np.triu(_as_adjacency(amat, geo.atom_count)))

    dist = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    rows = np.column_stack([np.minimum(zs[i], zs[j]), np.maximum(zs[i], zs[j]), dist])
    return _sort_rows(rows.reshape(-1, 3))


def angles(geo: Geometry, amat: ArrayLike) -> np.ndarray:
    """List bond angles.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        Sorted rows ``[z1, z2, z3, theta]`` with `theta` in radians, `z2` the
        central atom, and ``z1 <= z3``, shape ``(nangles, 4)``.

    Example:
        >>> geo = Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> a = angles(geo, adjacency_matrix(geo))
        >>> a.shape
        (1, 4)
        >>> round(float(a[0][-1]), 6)
        1.570796
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    nbrs = _neighbors(_as_adjacency(amat, geo.atom_count))

    triples = [
        (i, j, k)
        for j, nbr in enumerate(nbrs)
        for pos, i in enumerate(nbr)
        for k in nbr[pos + 1 :]
    ]
    i, j, k = np.array(triples, dtype=int).reshape(-1, 3).T

    v1 = xyz[i] - xyz[j]
    v2 = xyz[k] - xyz[j]
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos_theta = np.clip(np.einsum("nt,nt->n", v1, v2) / norms, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    rows = np.column_stack(
        [np.minimum(zs[i], zs[k]), zs[j], np.maximum(zs[i], zs[k]), theta]
    )
    return _sort_rows(rows.reshape(-1, 4))


def dihedrals(geo: Geometry, amat: ArrayLike) -> np.ndarray:
    """List dihedral angles.

    Args:
        geo: Geometry.
        amat: Adjacency matrix.

    Returns:
        Sorted rows ``[z1, z2, z3, z4, phi]`` with `phi` in radians, shape
        ``(ndihedrals, 5)``. Each row is oriented so that
        ``(z1, z2, z3, z4) <= (z4, z3, z2, z1)``, which leaves `phi` unchanged.

    Example:
        >>> geo = Geometry(
        ...     symbols=["H", "O", "O", "H"],
        ...     coordinates=[[0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> amat = np.array(
        ...     [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
        ... )
        >>> d = dihedrals(geo, amat)
        >>> d.shape
        (1, 5)
        >>> [float(x) for x in d[0][:4]]
        [1.0, 8.0, 8.0, 1.0]
        >>> round(float(d[0][-1]), 6)
        1.570796
    """
    zs = np.asarray(geo.atomic_numbers)
    xyz = geo.coordinates
    amat = _as_adjacency(amat, geo.atom_count)
    nbrs = _neighbors(amat)

    # Enumerate each dihedral once, about its unordered central bond (j, k)
    quads = [
        (i, j, k, l)
        for j, k in zip(*np.nonzero(np.triu(amat)), strict=True)
        for i in nbrs[j]
        if i != k
        for l in nbrs[k]  # noqa: E741
        if l not in (i, j)
    ]
    i, j, k, l = np.array(quads, dtype=int).reshape(-1, 4).T  # noqa: E741

    b1 = xyz[j] - xyz[i]
    b2 = xyz[k] - xyz[j]
    b3 = xyz[l] - xyz[k]

    nb2 = np.linalg.norm(b2, axis=1, keepdims=True)
    nb2 = np.where(nb2 == 0.0, 1.0, nb2)
    ub2 = b2 / nb2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    x = np.einsum("nt,nt->n", n1, n2)
    y = np.einsum("nt,nt->n", np.cross(n1, n2), ub2)
    phi = np.arctan2(y, x)

    zquads = zs[np.column_stack([i, j, k, l]).astype(int)].reshape(-1, 4)
    zquads = np.array(
        [min(tuple(zq), tuple(zq[::-1])) for zq in zquads.tolist()], dtype=float
    ).reshape(-1, 4)
    return _sort_rows(np.column_stack([zquads, phi]))


# Rigid-body transformations
def translate(
    geo: Geometry,
    arr: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Translate a geometry.

    Args:
        geo: Geometry.
        arr: Translation vector, or one vector per selected atom.
        keys: Atoms to translate. If `None`, translate all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The translated geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = np.add(geo.coordinates[mask], arr)
    return geo


def reflect(
    geo: Geometry,
    normal: ArrayLike,
    *,
    origin: ArrayLike = (0.0, 0.0, 0.0),
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Reflect a geometry across a plane.

    Args:
        geo: Geometry.
        normal: Normal vector of the reflection plane.
        origin: A point on the reflection plane.
        keys: Atoms to reflect. If `None`, reflect all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The reflected geometry.

    Raises:
        ValueError: If `normal` is a zero vector.
    """
    normal = np.asarray(normal, dtype=float)
    norm2 = np.dot(normal, normal)
    if not norm2 > 0:
        msg = f"Reflection plane normal must be a non-zero vector, got {normal}."
        raise ValueError(msg)

    geo = geo if in_place else geo.model_copy(deep=True)
    proj = np.outer(normal, normal) / norm2
    mask = slice(None) if keys is None else list(keys)
    origin = np.asarray(origin, dtype=float)
    coords = geo.coordinates[mask] - origin
    geo.coordinates[mask] = coords - 2 * coords @ proj + origin
    return geo


def rotate(
    geo: Geometry,
    rot: Rotation,
    *,
    origin: ArrayLike = (0.0, 0.0, 0.0),
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Rotate a geometry about a point.

    Args:
        geo: Geometry.
        rot: Rotation to apply.
        origin: Point to rotate about.
        keys: Atoms to rotate. If `None`, rotate all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The rotated geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    origin = np.asarray(origin, dtype=float)
    geo.coordinates[mask] = rot.apply(geo.coordinates[mask] - origin) + origin
    return geo


def twist(
    geo: Geometry, keys: Sequence[int], angle: float, *, degrees: bool = False
) -> Geometry:
    """Rotate a group about a bond, e.g. a methyl or amine group.

    Everything on the side of atom `j` of the bond ``(i, j)`` rotates about the
    axis from `i` to `j`, by the right-hand rule (thumb from `i` to `j`). Bond
    lengths and angles are kept; only the dihedral angles about the bond change.

    Args:
        geo: Geometry.
        keys: The bond, ``(i, j)``. The side of `j` moves.
        angle: Angle to rotate by.
        degrees: If `True`, `angle` is in degrees instead of radians.

    Returns:
        The twisted geometry.

    Raises:
        ValueError: If `i` and `j` aren't bonded, or the bond is in a ring.

    Example:
        >>> geo = Geometry(
        ...     symbols=["H", "O", "O", "H"],
        ...     coordinates=[[1, 0, 0], [0, 0, 0], [0, 0, 1.4], [1, 0, 1.4]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> geo = twist(geo, (1, 2), 90, degrees=True)
        >>> round(dihedral(geo, [0, 1, 2, 3], degrees=True), 6)
        90.0
    """
    i, j = _bond_keys(keys)
    moving = _bond_side(geo, i, j)
    axis = geo.coordinates[j] - geo.coordinates[i]
    angle = float(np.radians(angle)) if degrees else angle
    rot = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
    return rotate(geo, rot, origin=geo.coordinates[j], keys=moving)


def twist_angle(
    geo: Geometry,
    keys: Sequence[int],
    vec: ArrayLike,
    target: ArrayLike,
    *,
    degrees: bool = False,
) -> float:
    """Find the twist about a bond that turns one direction onto another.

    Both directions are projected onto the plane perpendicular to the bond, and
    the result is the signed angle between the projections, so that
    ``twist(geo, keys, twist_angle(geo, keys, vec, target))`` turns `vec` (as
    part of the moving group) to point as close as possible to `target`.

    Args:
        geo: Geometry.
        keys: The bond, ``(i, j)``, as for `twist`.
        vec: Direction attached to the moving group, e.g. a bond direction.
        target: Direction to turn it toward.
        degrees: If `True`, return the angle in degrees instead of radians.

    Returns:
        The angle, between -pi and pi (or -180 and 180 degrees).

    Raises:
        ValueError: If `vec` or `target` is parallel to the bond.
    """
    i, j = _bond_keys(keys)
    axis = geo.coordinates[j] - geo.coordinates[i]
    axis = axis / np.linalg.norm(axis)
    vec_perp, target_perp = (
        np.asarray(v, dtype=float) - np.dot(v, axis) * axis for v in (vec, target)
    )
    for name, v in (("vec", vec_perp), ("target", target_perp)):
        if np.isclose(np.linalg.norm(v), 0.0):
            msg = f"`{name}` is parallel to the bond {keys}."
            raise ValueError(msg)
    y = np.dot(np.cross(vec_perp, target_perp), axis)
    ang = float(np.arctan2(y, np.dot(vec_perp, target_perp)))
    return float(np.degrees(ang)) if degrees else ang


def _bond_keys(keys: Sequence[int]) -> tuple[int, int]:
    """Check that `keys` is a pair of distinct atoms."""
    if len(keys) != 2 or keys[0] == keys[1]:  # noqa: PLR2004
        msg = f"Expected a bond of two distinct atoms, got {keys}."
        raise ValueError(msg)
    return int(keys[0]), int(keys[1])


def _bond_side(geo: Geometry, i: int, j: int) -> list[int]:
    """List the atoms on the side of `j` of the bond ``(i, j)``, without `j`."""
    amat = adjacency_matrix(geo)
    if not amat[i, j]:
        msg = f"Atoms {i} and {j} are not bonded."
        raise ValueError(msg)
    amat[i, j] = amat[j, i] = 0
    _, labels = connected_components(amat, directed=False)
    if labels[i] == labels[j]:
        msg = f"The bond ({i}, {j}) is in a ring, so its sides can't twist."
        raise ValueError(msg)
    return [k for k in np.flatnonzero(labels == labels[j]).tolist() if k != j]


def center(geo: Geometry, *, in_place: bool = False) -> Geometry:
    """Translate a geometry so that its center of mass is at the origin.

    Args:
        geo: Geometry.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The centered geometry.
    """
    return translate(geo, -center_of_mass(geo), in_place=in_place)


def alignment_rotation(geo: Geometry, ref: Geometry) -> Rotation:
    """Find the rotation that best aligns a reference geometry onto a geometry.

    The rotation is found by the Kabsch algorithm, after moving both centers of
    mass to the origin. Atoms are matched by index.

    Args:
        geo: Target geometry.
        ref: Reference geometry, with the same atoms as `geo`.

    Returns:
        The rotation `rot` minimizing the RMSD between ``rotate(center(ref), rot)``
        and ``center(geo)``.

    Raises:
        ValueError: If the geometries have different symbols.

    Example:
        >>> ref = Geometry(
        ...     symbols=["O", "H", "H"],
        ...     coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     charge=0,
        ...     spin=0,
        ... )
        >>> rot = Rotation.from_euler("z", 30, degrees=True)
        >>> geo = rotate(ref, rot)
        >>> round(float(alignment_rotation(geo, ref).magnitude()), 6)
        0.523599
    """
    if geo.symbols != ref.symbols:
        msg = f"Symbols do not match: {geo.symbols} != {ref.symbols}"
        raise ValueError(msg)

    # The second return value is the root-sum-square distance
    return Rotation.align_vectors(
        center(geo).coordinates, center(ref).coordinates, weights=geo.masses
    )[0]


def axis_alignment_rotation(
    geo: Geometry, keys: Sequence[int], axis: ArrayLike = (0.0, 0.0, 1.0)
) -> Rotation:
    """Find the smallest rotation pointing the vector between two atoms along an axis.

    Use it to put a bond or other molecular axis along a coordinate axis, e.g.
    before spinning the geometry about that axis.

    Args:
        geo: Geometry.
        keys: Two atoms, ``(i, j)``. The vector from `i` to `j` is aligned.
        axis: Direction to point it in.

    Returns:
        The rotation, to apply with `rotate` (about atom `i` to keep it in
        place, e.g. ``rotate(geo, rot, origin=geo.coordinates[i])``).

    Raises:
        ValueError: If `keys` isn't two distinct atoms at different positions.

    Example:
        >>> geo = Geometry(
        ...     symbols=["H", "H"], coordinates=[[0, 0, 0], [1, 0, 0]], charge=0, spin=0
        ... )
        >>> rot = axis_alignment_rotation(geo, [0, 1], axis=[0, 1, 0])
        >>> rotate(geo, rot).coordinates.round(6).tolist()
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """
    if len(keys) != 2 or keys[0] == keys[1]:  # noqa: PLR2004
        msg = f"Expected two distinct atoms, got {keys}."
        raise ValueError(msg)
    i, j = keys
    vec = geo.coordinates[j] - geo.coordinates[i]
    if np.isclose(np.linalg.norm(vec), 0.0):
        msg = f"Atoms {i} and {j} are at the same position."
        raise ValueError(msg)
    # For a single pair of vectors, scipy returns the smallest rotation
    return Rotation.align_vectors([axis], [vec])[0]


# Visualization
class View(py3Dmol.view):
    """A `py3Dmol.view` with helpers for geometries and arrows."""

    def add_geometry(self, geo: Geometry, *, label: bool = False) -> None:
        """Add a geometry to the view.

        Args:
            geo: Geometry.
            label: If `True`, label atoms by index.
        """
        view(geo, view=self, label=label)

    def add_xyz_axes(
        self,
        *,
        scale: float = 1,
        colors: tuple[str, str, str] = ("red", "green", "blue"),
    ) -> None:
        """Add x, y, and z axis arrows from the origin.

        Args:
            scale: Length of each arrow.
            colors: Colors of the x, y, and z arrows.
        """
        self.add_vectors(np.eye(3) * scale, colors=colors)

    def add_vectors(
        self,
        coords: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        colors: Sequence[str] | None = None,
    ) -> None:
        """Add arrows that share a start point.

        Args:
            coords: Arrow tips, one row per arrow.
            start_coord: Start point shared by all arrows.
            direction: If `True`, treat each row of `coords` as a direction
                from `start_coord` rather than an end point.
            colors: One color per arrow. Defaults to black.

        Raises:
            ValueError: If the numbers of arrows and colors differ.
        """
        coords = np.asarray(coords, dtype=np.float64)
        colors = colors or ["black"] * len(coords)
        if len(coords) != len(colors):
            msg = f"Coordinates and colors do not match: {coords = }, {colors = }"
            raise ValueError(msg)

        for coord, color in zip(coords, colors, strict=True):
            self.add_vector(coord, start_coord, direction=direction, color=color)

    def add_vector(
        self,
        coord: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        color: str = "black",
    ) -> None:
        """Add an arrow.

        Args:
            coord: Arrow tip.
            start_coord: Arrow start point.
            direction: If `True`, treat `coord` as a direction from
                `start_coord` rather than an end point.
            color: Arrow color.
        """
        if direction:
            coord = np.add(coord, start_coord)

        start = np.asarray(start_coord).tolist()
        end = np.asarray(coord).tolist()
        self.addArrow(
            {
                "start": {"x": start[0], "y": start[1], "z": start[2]},
                "end": {"x": end[0], "y": end[1], "z": end[2]},
                "color": color,
            }
        )


def view(
    geo: Geometry, *, view: py3Dmol.view | None = None, label: bool = False
) -> py3Dmol.view:
    """Display a geometry with py3Dmol.

    Args:
        geo: Geometry.
        view: Existing view to add the geometry to. If `None`, create one.
        label: If `True`, label atoms by index.

    Returns:
        The view containing the geometry.
    """
    view = py3Dmol.view(width=400, height=400) if view is None else view
    view.addModel(geo.xyz_block(), "xyz")
    # Model -1 is the one just added, so earlier models keep their styling
    view.setStyle({"model": -1}, {"stick": {}, "sphere": {"scale": 0.3}})
    if label:
        for key in range(geo.atom_count):
            view.addLabel(
                str(key),
                {
                    "backgroundOpacity": 0.0,
                    "fontColor": "black",
                    "alignment": "center",
                    "inFront": True,
                },
                {"model": -1, "index": key},
            )
    return view


def render_svg(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: "str | xyzrender.RenderConfig" = "default",
    include_h: bool = True,
) -> "xyzrender.SVGResult":
    """Render a geometry as an SVG image with xyzrender.

    The result displays inline in Jupyter.

    Args:
        geo: Geometry.
        out: Output path. The suffix is replaced with ``.svg``.
        config: xyzrender preset name or `xyzrender.RenderConfig`.
        include_h: If `True`, include hydrogen atoms.

    Returns:
        The SVG render result.
    """
    xyzrender = _import_xyzrender()
    out = Path(out).with_suffix(".svg") if out else out
    mol = _xyzrender_molecule(geo)
    return xyzrender.render(mol, config=config, hy=include_h, output=out)


def render_gif(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: "str | xyzrender.RenderConfig" = "default",
    include_h: bool = True,
    rotation_axis: str = "x",
) -> "xyzrender.GIFResult":
    """Render a geometry rotating about an axis as a GIF with xyzrender.

    The result displays inline in Jupyter.

    Args:
        geo: Geometry.
        out: Output path. The suffix is replaced with ``.gif``.
        config: xyzrender preset name or `xyzrender.RenderConfig`.
        include_h: If `True`, include hydrogen atoms.
        rotation_axis: Axis to rotate about.

    Returns:
        The GIF render result.
    """
    xyzrender = _import_xyzrender()
    out = Path(out).with_suffix(".gif") if out else out
    mol = _xyzrender_molecule(geo)
    return xyzrender.render_gif(
        mol, config=config, hy=include_h, output=out, gif_rot=rotation_axis
    )


def _import_xyzrender() -> ModuleType:
    """Import xyzrender, which is optional for conda installs (PyPI-only)."""
    try:
        import xyzrender  # noqa: PLC0415
    except ImportError as err:
        msg = (
            "Rendering requires xyzrender, which is only on PyPI. "
            "Install it with `pip install xyzrender` or, in a pixi workspace, "
            "add it under [pypi-dependencies]."
        )
        raise ImportError(msg) from err
    return xyzrender


def _xyzrender_molecule(geo: Geometry) -> "xyzrender.Molecule":
    """Load a geometry into xyzrender via a temporary xyz file."""
    xyzrender = _import_xyzrender()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "geometry.xyz"
        xyz_file(geo, path=path)
        return xyzrender.load(path)


# Helpers
def _key_coordinates(geo: Geometry, keys: Sequence[int], count: int) -> np.ndarray:
    """Get the coordinates of `count` atoms, checking the number of keys."""
    keys = list(keys)
    if len(keys) != count:
        msg = f"Expected {count} atom indices, got {keys}."
        raise ValueError(msg)
    return geo.coordinates[keys]


def _as_adjacency(amat: ArrayLike, natms: int) -> np.ndarray:
    """Convert to a boolean adjacency matrix, checking its shape."""
    amat = np.asarray(amat).astype(bool)
    if amat.shape != (natms, natms):
        msg = f"Expected adjacency matrix of shape {(natms, natms)}, got {amat.shape}."
        raise ValueError(msg)
    return amat


def _neighbors(amat: np.ndarray) -> list[list[int]]:
    """Get the sorted neighbor indices of each atom."""
    return [np.flatnonzero(row).tolist() for row in amat]


def _sort_rows(arr: np.ndarray) -> np.ndarray:
    """Sort the rows of a 2D array lexicographically."""
    return arr[np.lexsort(arr.T[::-1])]
