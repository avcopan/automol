"""The `Geometry` model and the algorithms that operate on it."""

import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import py3Dmol
import pyparsing as pp
import xyzrender
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

__all__ = [
    "Geometry",
    "View",
    "adjacency_matrix",
    "angles",
    "bonds",
    "center_of_mass",
    "dihedrals",
    "distance_keys",
    "distance_matrix",
    "from_rdkit_mol",
    "from_xyz_block",
    "from_xyz_file",
    "inertia_tensor",
    "rdkit_mol",
    "reflect",
    "render_gif",
    "render_svg",
    "rotate",
    "translate",
    "view",
    "xyz_block",
    "xyz_file",
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
    for sym, (x, y, z) in zip(geo.symbols, geo.coordinates, strict=True):
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
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Reflect a geometry across a plane through the origin.

    Args:
        geo: Geometry.
        normal: Normal vector of the reflection plane.
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
    geo.coordinates[mask] = geo.coordinates[mask] - 2 * geo.coordinates[mask] @ proj
    return geo


def rotate(
    geo: Geometry,
    rot: Rotation,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Rotate a geometry about the origin.

    Args:
        geo: Geometry.
        rot: Rotation to apply.
        keys: Atoms to rotate. If `None`, rotate all atoms.
        in_place: If `True`, modify `geo` instead of returning a copy.

    Returns:
        The rotated geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = rot.apply(geo.coordinates[mask])
    return geo


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
    config: str | xyzrender.RenderConfig = "default",
    include_h: bool = True,
) -> xyzrender.SVGResult:
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
    out = Path(out).with_suffix(".svg") if out else out
    mol = _xyzrender_molecule(geo)
    return xyzrender.render(mol, config=config, hy=include_h, output=out)


def render_gif(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: str | xyzrender.RenderConfig = "default",
    include_h: bool = True,
    rotation_axis: str = "x",
) -> xyzrender.GIFResult:
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
    out = Path(out).with_suffix(".gif") if out else out
    mol = _xyzrender_molecule(geo)
    return xyzrender.render_gif(
        mol, config=config, hy=include_h, output=out, gif_rot=rotation_axis
    )


def _xyzrender_molecule(geo: Geometry) -> xyzrender.Molecule:
    """Load a geometry into xyzrender via a temporary xyz file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "geometry.xyz"
        xyz_file(geo, path=path)
        return xyzrender.load(path)


# Helpers
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
