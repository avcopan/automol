"""Molecular geometry functions."""

import contextlib
import hashlib
import itertools
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import py3Dmol
import pyparsing as pp
import xyzrender
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, model_validator
from pyparsing import pyparsing_common as ppc
from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds
from scipy import spatial
from scipy.spatial.transform import Rotation

from . import element, rd
from .utils import constants
from .utils.exc import GeometryConversionError, HashGenerationError, XYZFormatError
from .utils.types import CoordinatesField, FloatArray

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbols: list[str]
    coordinates: CoordinatesField
    charge: int | None
    spin: int | None

    hash: str | None = None

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
    def from_xyz_block(cls, xyz_block: str, *, charge: int, spin: int) -> "Geometry":
        """Instantiate Geometry from a formatted xyz block."""
        return from_xyz_block(xyz_block, charge=charge, spin=spin)

    def xyz_file(self, *, path: str | Path, comment: str | None = None) -> None:
        """Write Geometry as a formatted xyz file with optional comment."""
        xyz_file(self, path=path, comment=comment)

    @classmethod
    def from_xyz_file(cls, path: str | Path, *, charge: int, spin: int) -> "Geometry":
        """Instantiate Geometry from a formatted xyz file."""
        return from_xyz_file(path, charge=charge, spin=spin)

    @model_validator(mode="after")
    def populate_hash(self) -> Self:
        """Populate hash after model is validated."""
        # Only populate if hash wasn't explicitly provided
        if self.hash is None:
            with contextlib.suppress(HashGenerationError):
                self.hash = geometry_hash(self, decimals=6)
                self.model_fields_set.add("hash")
        return self


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
    if geo.spin is None:
        msg = "Cannot determine bond connectivity without an assigned spin."
        raise GeometryConversionError(msg)

    if geo.charge is None:
        msg = "Cannot determine bond connectivity without an assigned charge."
        raise GeometryConversionError(msg)

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


# Properties
def geometry_hash(geo: Geometry, decimals: int = 6) -> str:
    """Generate a deterministic geometry hash string.

    Parameters
    ----------
    decimals
        Number of decimal places to round the coordinates before hashing.

    Returns
    -------
        Geometry hash string.
    """
    # Check that all hash fields are present
    if geo.charge is None or geo.spin is None:
        msg = "Geometry charge and spin must be present for hashing."
        raise HashGenerationError(msg, geo)
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
    xyz_str = geo.xyz_block()
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


def render_svg(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: str | xyzrender.RenderConfig = "default",
    include_h: bool = True,
) -> xyzrender.SVGResult:
    """Render geometry in .svg format.

    Results display inlay automatically.

    Parameters
    ----------
    geo
        Geometry.
    out
        Output path for rendered image.
    config
        xyzrender RenderConfig settings.
    include_h
        If True, include hydrogen atoms in render.

    Returns
    -------
    SVGResult
    """
    out = Path(out).with_suffix(".svg") if out else out

    tmp_file = Path.cwd() / ".tmp.xyz"
    xyz_file(geo, path=tmp_file)
    mol = xyzrender.load(tmp_file)

    tmp_file.unlink()
    return xyzrender.render(mol, config=config, hy=include_h, output=out)


def render_gif(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: str | xyzrender.RenderConfig = "default",
    include_h: bool = True,
    rotation_axis: str = "x",
) -> xyzrender.GIFResult:
    """Render geometry rotating about an axis in .gif format.

    Results display inlay automatically.

    Parameters
    ----------
    geo
        Geometry.
    out
        Output path for rendered gif.
    config
        xyzrender RenderConfig settings.
    include_h
        If True, include hydrogen atoms in render.
    rotation_axis
        Axis to rotate about in animation.

    Returns
    -------
    GIFResult
    """
    out = Path(out).with_suffix(".gif") if out else out

    tmp_file = Path.cwd() / ".tmp.xyz"
    xyz_file(geo, path=tmp_file)
    mol = xyzrender.load(tmp_file)

    tmp_file.unlink()
    return xyzrender.render_gif(
        mol, config=config, hy=include_h, output=out, gif_rot=rotation_axis
    )


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
    return spatial.distance_matrix(geo.coordinates, geo.coordinates)


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

    z = r23 / np.linalg.norm(r23)
    y = n123 / np.linalg.norm(n123)
    x = np.cross(y, z)

    # Determine components of 3-4 bond along x and y and calculate angle from arctan
    v = r34 / np.linalg.norm(r34)
    vx = np.dot(v, x)
    vy = np.dot(v, y)
    angle = np.arctan2(vy, vx)
    return angle * constants.RADIANS_TO_DEGREES if degrees else angle


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


def mass_weight_vector(geo: Geometry) -> np.ndarray:
    """Get the mass-weighting vector of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry

    Returns
    -------
    mass-weighting vector
    """
    return np.sqrt(np.repeat(geo.masses, 3))


def rotational_analysis(geo: Geometry) -> tuple[FloatArray, FloatArray]:
    """Calculate rotational analysis of a geometry.

    Parameters
    ----------
    geo
        Geometry.

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


def translational_normal_modes(
    geo: Geometry, *, mass_weighted: bool = True
) -> np.ndarray:
    """Calculate translational normal modes of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry
    mass_weighted
        If True, return mass-weighted normal modes.
    """
    trans_coos = np.tile(np.eye(3), (geo.atom_count, 1))

    if mass_weighted:
        trans_coos *= np.sqrt(mass_weight_vector(geo))[:, np.newaxis]

    return trans_coos


def rotational_normal_modes(geo: Geometry, *, mass_weighted: bool = True) -> np.ndarray:
    """Calculate rotational normal modes of a geometry.

    Parameters
    ----------
    geo
        Instance of a Geometry
    mass_weighted
        If True, return mass-weighted normal modes.
    """
    _, rot_axes = rotational_analysis(geo)
    coords = (geo.coordinates - center_of_mass(geo)) * constants.ANGSTROM_TO_BOHR
    rot_coos = [
        np.concatenate([np.cross(xyz, rot_axis) for xyz in coords])
        for rot_axis in rot_axes
    ]
    rot_coos = np.transpose(rot_coos)

    if mass_weighted:
        rot_coos *= np.sqrt(mass_weight_vector(geo))[:, np.newaxis]

    return rot_coos


def normal_mode_projection(
    geo: Geometry, *, trans: bool = False, rot: bool = False
) -> np.ndarray:
    """Get the matrix for projecting onto a subset of normal modes.

    Parameters
    ----------
    geo
        Instance of a Geometry.
    trans
        If True, keep translational modes.
    rot
        If True, keep rotational modes.
    """
    coos = []
    if not trans:
        coos.append(translational_normal_modes(geo, mass_weighted=True))

    if not rot:
        coos.append(rotational_normal_modes(geo, mass_weighted=True))

    if not coos:
        return np.eye(geo.atom_count * 3)

    coos = np.hstack(coos)
    dim = np.shape(coos)[-1]
    coo_basis, *_ = np.linalg.svd(coos, full_matrices=True)

    return coo_basis[:, dim:]


def vibrational_analysis(
    geo: Geometry, hess: list[list[float]], *, trans: bool = False, rot: bool = False
) -> tuple[tuple[float, ...], npt.NDArray]:
    """Calculate frequencies and vibrational modes from a Hessian matrix.

    Parameters
    ----------
    geo
        Instance of a Geometry
    hess
        `(3N, 3N)` array of hessian values.
    trans
        If True, keep translational modes.
    rot
        If True, keep rotational modes.

    Returns
    -------
    frequencies
        Vibrational frequencies.
    modes
        Vibrational modes.

    Raises
    ------
    ValueError
        `Hessian shape is not (3N, 3N)`.
    """
    exp_dim = 3 * geo.atom_count
    if len(hess) != exp_dim or len(hess[0]) != exp_dim:
        msg = f"Hessian shape ({len(hess)}, {len(hess[0])}) is not (3N, 3N)."
        raise ValueError(msg)

    masses = mass_weight_vector(geo)
    hess_mw = hess / np.outer(masses, masses)

    proj = normal_mode_projection(geo, trans=trans, rot=rot)
    hess_proj = proj.T @ hess_mw @ proj

    evals, evecs = np.linalg.eigh(hess_proj)

    norm_coos = np.dot(proj, evecs) / masses[:, np.newaxis]
    norm_coos /= np.linalg.norm(norm_coos, axis=0)

    freqs = (
        np.sqrt(np.complex128(evals)) * constants.VIBRATIONAL_FORCE_TO_INV_CM_FREQUENCY
    )
    freqs = tuple(map(float, np.real(freqs) - np.imag(freqs)))

    return freqs, norm_coos


def harmonic_zpv(
    geo: Geometry, hess: list[list[float]], *, freqs: tuple[float, ...] | None = None
) -> float:
    """Calculate the harmonic zero point vibrational energy of a geometry."""
    if freqs is None:
        freqs, _ = vibrational_analysis(geo, hess)

    zpe_wavenumbers = 0.5 * sum([f for f in freqs if f > 0.0])

    return zpe_wavenumbers * constants.WAVENUMBER_TO_HARTREE


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
