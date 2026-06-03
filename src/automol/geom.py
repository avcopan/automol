"""Molecular geometries."""

import functools
import hashlib
import itertools
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import numpy as np
import pint
import py3Dmol
import pyparsing as pp
import scipy
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyparsing import pyparsing_common as ppc
from qcdata import Structure
from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds
from scipy.spatial.transform import Rotation

from . import element, rd
from .types import CoordinatesField, FloatArray

try:
    from qcdata import Structure

    _HAS_QC = True

except ImportError:
    _HAS_QC = False


RADIANS_TO_DEGREES = pint.Quantity("radian").m_as("degree")
DEGREES_TO_RADIANS = 1 / RADIANS_TO_DEGREES


def requires_qcdata[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Raise if qcdata is unavailable."""

    @functools.wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not _HAS_QC:
            msg = f"Function '{func.__name__}' requires qcdata."  # ty:ignore[unresolved-attribute]
            raise RuntimeError(msg)
        return func(*args, **kwargs)

    return _wrapper


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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int = 0
    spin: int = 0

    hash: str | None = Field(default=None)

    @property
    def masses(self) -> list[float]:
        """Get isotopic masses."""
        return list(map(element.mass, self.symbols))

    @property
    def atomic_numbers(self) -> list[float]:
        """Get atomic numbers."""
        return list(map(element.number, self.symbols))

    @property
    def covalent_radii(self) -> list[float]:
        """Get Pyykko covalent radii in A."""
        return list(map(element.covalent_radius, self.symbols))

    @property
    def nvalences(self) -> list[int]:
        """Get numbers of valence electrons."""
        return list(map(element.nvalence, self.symbols))

    @model_validator(mode="after")
    def populate_hash(self) -> "Geometry":
        """Populate hash immediately after the model is created."""
        # Only populate if hash wasn't explicitly provided
        if self.hash is None:
            self.hash = geometry_hash(self, decimals=6)
        return self


# File I/O
def read_xyz_file(path: str | Path) -> Geometry:
    """Read a geometry from an XYZ file.

    Parameters
    ----------
    path
        Path to XYZ file.

    Returns
    -------
        Geometry.
    """
    path = path if isinstance(path, Path) else Path(path)
    return from_xyz_block(path.read_text())


def write_xyz_file(geo: Geometry, path: str | Path) -> None:
    """Write a geometry to an XYZ file.

    Parameters
    ----------
    geo
        Geometry.
    path
        Path to XYZ file.
    """
    path = path if isinstance(path, Path) else Path(path)
    path.write_text(xyz_block(geo))


# Importers / Exporters
def xyz_block(geo: Geometry) -> str:
    """
    Return geometry as formatted xyz block.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    lines = [f"{len(geo.symbols)}", ""]
    for sym, (x, y, z) in zip(geo.symbols, geo.coordinates, strict=True):
        lines.append(f"{sym:<2} {x:12.8f} {y:12.8f} {z:12.8f}")

    return "\n".join(lines)


CHAR = pp.Char(pp.alphas)
SYMBOL = pp.Combine(CHAR + pp.Opt(CHAR))
XYZ_LINE = SYMBOL + pp.Group(ppc.fnumber * 3) + pp.Suppress(... + pp.LineEnd())


def from_xyz_block(xyz_str: str) -> Geometry:
    """
    Instantiate Geometry from formatted xyz block.

    Parameters
    ----------
    geo_str
        Formatted xyz block.

    Returns
    -------
    Geometry
        Geometry object.
    """
    xyz_str = xyz_str.strip()
    lines = xyz_str.splitlines()[2:]
    if not lines:
        return Geometry(symbols=[], coordinates=[])  # ty:ignore[invalid-argument-type]

    symbs, coords = zip(
        *[XYZ_LINE.parse_string(line).as_list() for line in lines], strict=True
    )
    return Geometry(symbols=list(symbs), coordinates=np.array(coords))


def rdkit_mol(geo: Geometry) -> Mol:
    """
    Instantiate an rdkit Mol from a Geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    Mol
        rdkit Mol instance.
    """
    if geo.charge != 0:
        msg = f"Determining bond connectivity with charges not implemented.\n{geo = }"
        raise NotImplementedError(msg)

    raw_mol = Chem.MolFromXYZBlock(xyz_block(geo))
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(conn_mol, charge=-geo.spin)

    for a in conn_mol.GetAtoms():
        charge = a.GetFormalCharge()
        a.SetNumRadicalElectrons(abs(charge))
        a.SetFormalCharge(0)

    return conn_mol


def from_rdkit_mol(mol: Mol) -> Geometry:
    """
    Generate geometry from RDKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
        Geometry.
    """
    if not rd.mol.has_coordinates(mol):
        mol = rd.mol.add_coordinates(mol)

    return Geometry(
        symbols=rd.mol.symbols(mol),
        coordinates=rd.mol.coordinates(mol),
        charge=rd.mol.charge(mol),
        spin=rd.mol.spin(mol),
    )


@requires_qcdata
def qc_structure(geo: Geometry) -> Structure:
    """
    Instantiate a qc Structure from a Geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    Structure
    """
    return Structure(
        symbols=geo.symbols,
        geometry=np.array(geo.coordinates) * pint.Quantity("angstrom").m_as("bohr"),
        charge=geo.charge,
        multiplicity=geo.spin + 1,
    )


@requires_qcdata
def from_qc_structure(struc: Structure) -> Geometry:
    """
    Instantiate Geometry from qc Structure.

    Parameters
    ----------
    struc
        qc Structure.

    Returns
    -------
    Geometry
    """
    return Geometry(
        symbols=struc.symbols,
        coordinates=struc.geometry * pint.Quantity("bohr").m_as("angstrom"),
        charge=struc.charge,
        spin=struc.multiplicity - 1,
    )


def smiles(geo: Geometry) -> str:
    """
    Provide SMILES string from Geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    smiles
        SMILES formatted string.
    """
    mol = rdkit_mol(geo)
    return rd.mol.smiles(mol)


def from_smiles(smi: str) -> Geometry:
    """
    Instantiate Geometry from SMILES string.

    Parameters
    ----------
    smi
        SMILES formatted string.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    mol = rd.mol.from_smiles(smi)
    return from_rdkit_mol(mol)


def inchi(geo: Geometry) -> str:
    """
    Provide InChI string from Geometry through RDKit.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    inchi
        InChI formatted string.
    """
    mol = rdkit_mol(geo)
    return rd.mol.inchi(mol)


def from_inchi(inchi: str) -> Geometry:
    """
    Instantiate Geometry from InChI string.

    Parameters
    ----------
    smi
        SMILES formatted string.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    mol = rd.mol.from_inchi(inchi)
    return from_rdkit_mol(mol)


# Properties
def geometry_hash(geo: Geometry, decimals: int = 6) -> str:
    """
    Generate geometry hash string.

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


def center_of_mass(geo: Geometry) -> FloatArray:
    """
    Calculate geometry center of mass.

    Parameters
    ----------
        Geometry.

    Returns
    -------
        Center of mass coordinates.
    """
    masses = list(map(element.mass, geo.symbols))
    coords = geo.coordinates
    return np.sum(np.reshape(masses, (-1, 1)) * coords, axis=0) / np.sum(masses)


def distance_matrix(geo: Geometry) -> FloatArray:
    """
    Compute the distance matrix for a geometry.

    Parameters
    ----------
    geo
        Geometry object.

    Returns
    -------
    FloatArray
        Distance matrix of geometry.
    """
    return scipy.spatial.distance_matrix(geo.coordinates, geo.coordinates)


def dihedral_angle(
    geo: Geometry, keys: Sequence[int], *, degrees: bool = True
) -> float:
    """Calculate the dihedral angle defined by four atoms.

    Parameters
    ----------
    geo
        Geometry.
    keys
        Indices of the four atoms defining the dihedral angle.
    degrees
        Whether to return the angle in degrees or radians.

    Returns
    -------
        Dihedral angle.
    """
    coords = geo.coordinates[list(keys)]
    if len(coords) != 4:  # noqa: PLR2004
        msg = "Exactly four atoms must be specified for dihedral angle."
        raise ValueError(msg)

    # Determine bond vectors and 1-2-3 plane normal
    r1, r2, r3, r4 = coords
    r12 = r2 - r1
    r23 = r3 - r2
    r34 = r4 - r3
    n123 = np.cross(r12, r23)

    # Form coordinate system with x upward in plane, y along plane normal, and z
    # away along central bond:
    #
    #     x
    #     ^
    #     1
    #     |
    #     2/3   > y
    #      \
    #       4
    #
    z = r23 / np.linalg.norm(r23)
    y = n123 / np.linalg.norm(n123)
    x = np.cross(y, z)

    # Determine components of 3-4 bond along x and y and calculate angle from arctan
    v = r34 / np.linalg.norm(r34)
    vx = np.dot(v, x)
    vy = np.dot(v, y)
    angle = np.arctan2(vy, vx)
    return angle * RADIANS_TO_DEGREES if degrees else angle


def inertia_tensor(geo: Geometry) -> FloatArray:
    """Calculate the inertia tensor of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia tensor.
    """
    masses = geo.masses
    coords = geo.coordinates - center_of_mass(geo)
    return sum(
        m * (np.vdot(r, r) * np.eye(3) - np.outer(r, r))
        for (r, m) in zip(coords, masses, strict=True)
    )


def inertia_moments(geo: Geometry) -> FloatArray:
    """Calculate the moments of inertia of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia moments.
    """
    evals, *_ = rotational_analysis(geo)
    return evals


def inertia_axes(geo: Geometry) -> FloatArray:
    """Calculate the axes of inertia of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia axes.
    """
    _, evecs = rotational_analysis(geo)
    return evecs


def rotation_to_inertia_axes(geo: Geometry) -> Rotation:
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


def rotational_analysis(geo: Geometry) -> tuple[FloatArray, FloatArray]:
    """Calculate rotational analysis of a geometry.

    Parameters
    ----------
    geo
        Geometry.
    drop_null
        Whether to drop null eigenvalues.

    Returns
    -------
        Eigenvalues and eigenvectors of the inertia tensor.
    """
    inert = inertia_tensor(geo)
    evals, evecs = np.linalg.eigh(inert)
    # Ensure right-handed coordinate system
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1  # flip one eigenvector
    return evals, evecs


# Comparison
def kabsch(
    geo1: Geometry, geo2: Geometry, *, heavy_only: bool = False
) -> tuple[FloatArray, FloatArray, float]:
    """
    Compute the optimal rotation / translation to align two Geometries and their RMSD.

    For more information on the numerical method, see https://hunterheidenreich.com/posts/kabsch-algorithm/

    Parameters
    ----------
    geo1
        Geometry object.
    geo2
        Geometry object.
    heavy_only
        If True, only consider heavy atoms.

    Returns
    -------
    FloatArray
        Optimal rotation of geo2 onto geo1
    FloatArray
        Optimal translation of geo2 onto geo1
    float
        RMSD
    """
    p = np.array(geo1.coordinates)
    q = np.array(geo2.coordinates)
    p_masses = geo1.masses
    q_masses = geo2.masses

    if heavy_only:
        mask_p = np.array([s != "H" for s in geo1.symbols])
        mask_q = np.array([s != "H" for s in geo2.symbols])
        # Contrapositive of "If no heavy atoms exist (e.g., H2, H), skip masking"
        if np.any(mask_p):
            p, q = p[mask_p], q[mask_q]

            p_masses = np.asanyarray(p_masses)
            q_masses = np.asanyarray(q_masses)

            p_masses, q_masses = p_masses[mask_p], q_masses[mask_q]

    if p.shape != q.shape:
        msg = f"""
        Input arrays must have same number of dimensions.\n
        {p.shape = }\n
        {q.shape = }\n
        """
        raise ValueError(msg)

    # --- Optimal translation -------------------
    centroid_p = center_of_mass(geo1)
    centroid_q = center_of_mass(geo2)
    t = centroid_p - centroid_q  # Optimal translation
    # Center the coordinates
    p = p - centroid_p
    q = q - centroid_q

    # --- Optimal rotation ----------------------
    H = np.dot(p.T, q)  # Covariance matrix  # noqa: N806
    U, _, Vt = np.linalg.svd(H)  # noqa: N806

    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:  # Validate right-handed coordinates
        Vt[-1, :] *= -1.0

    R = np.dot(Vt.T, U.T)  # Optimal rotation  # noqa: N806

    # --- RMSD ----------------------------------
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / p.shape[0])

    return R, t, rmsd


def is_similar(
    geo1: Geometry,
    geo2: Geometry,
    *,
    rmsd_tol: float = 1e-1,
) -> bool:
    """
    Determine whether two geometries are similar.

    Parameters
    ----------
    geo1
        Geometry object.
    geo2
        Geometry object.
    rmsd_tol
        Maximum allowed RMSD.

    Returns
    -------
    bool
        Whether the two geometries are similar.
    """
    # --- InChI    ---
    if inchi(geo1) != inchi(geo2):
        return False

    # --- Geometry Hash ---
    if geometry_hash(geo1) == geometry_hash(geo2):
        return True

    # --- Heavy Atom RMSD ---
    if geo1.symbols != geo2.symbols:
        msg = "Atomic symbols do not map onto each other. RMSD cannot be computed."
        raise ValueError(msg)

    msg = "Not implemented until canonical ordering is established."
    raise NotImplementedError(msg)

    _, _, rmsd = kabsch(geo1, geo2, heavy_only=True)

    return rmsd < rmsd_tol


# Transformation
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


def to_eckart_frame(geo: Geometry, *, in_place: bool = False) -> Geometry:
    """Rotate geometry to align with inertia axes.

    Parameters
    ----------
    geo
        Geometry.
    in_place
        Whether to rotate in place or return a new geometry.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    # Move to center of mass
    geo = translate(geo, -center_of_mass(geo), in_place=True)
    # Rotate to inertia axes
    rot = rotation_to_inertia_axes(geo)
    return rotate(geo, rot, in_place=True)


def set_distance(
    geo: Geometry,
    *,
    idxs: Sequence[int],
    val: float,
    max_change: float = 0.25,
    in_place: bool = False,
) -> Geometry:
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


# Multi-geometry operations
def concat(geos: Sequence[Geometry]) -> Geometry:
    """Concatenate geometries.

    Parameters
    ----------
    geos
        List of geometries.

    Returns
    -------
        Geometry.
    """
    symbols = list(itertools.chain.from_iterable(geo.symbols for geo in geos))
    coordinates = np.vstack([geo.coordinates for geo in geos])
    charge = sum(geo.charge for geo in geos)
    spin = sum(geo.spin for geo in geos)
    return Geometry(symbols=symbols, coordinates=coordinates, charge=charge, spin=spin)


# Visualization
def view(
    geo: Geometry, *, view: py3Dmol.view | None = None, label: bool = False
) -> py3Dmol.view:
    """View a geometry with py3Dmol.

    Parameters
    ----------
    geo
        Geometry.
    view
        py3Dmol view.
    label
        Whether to add atom labels to the view.

    Returns
    -------
        py3Dmol view.
    """
    view = py3Dmol.view(width=400, height=400) if view is None else view
    xyz_str = xyz_block(geo)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    if label:
        for key in range(len(geo.symbols)):
            view.addLabel(
                key,
                {
                    "backgroundOpacity": 0.0,
                    "fontColor": "black",
                    "alignment": "center",
                    "inFront": True,
                },
                {"index": key},
            )
    return view


def adjacency_matrix(
    geo: Geometry,
    *,
    delta: float = 0.5,
    override_bonding_capacities: dict[str, int] | None = None,
) -> ArrayLike:
    """Determine neighboring atoms."""
    dmat = distance_matrix(geo)
    radii = np.array(geo.covalent_radii)

    r_cov_matrix = radii[:, np.newaxis] + radii[np.newaxis, :] + delta
    amat = dmat <= r_cov_matrix
    np.fill_diagonal(amat, 0)  # Atoms don't neighbor themselves

    caps = [
        element.bonding_capacity(s, override=override_bonding_capacities)
        for s in geo.symbols
    ]
    vals = np.sum(amat, axis=0)

    for i, (symb, val, cap) in enumerate(zip(geo.symbols, vals, caps, strict=True)):
        if val > cap:
            msg = f"Atom {symb}:{i} degree ({val}) exceeds bonding capacity ({cap})."
            raise NotImplementedError(msg)

    return amat
