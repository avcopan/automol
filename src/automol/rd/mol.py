"""RDKit molecule interface."""

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdDetermineBonds
from rdkit.Chem.rdDistGeom import EmbedMolecule
from scipy.optimize import Bounds, LinearConstraint, milp

from ..utils import element
from ..utils.exc import GeometryConversionError
from ..utils.types import FloatArray

# Penalties for the bond perception integer program in `from_connectivity`.
# Charges cost more than valence expansion so that, e.g., DMSO is CS(C)=O rather
# than C[S+](C)[O-], and a closed-shell lone pair (singlet carbene) costs more than
# a pair of charges so that CO is [C-]#[O+].
_CHARGE_COST = 10.0
_EXPANDED_VALENCE_COST = 4.0
_LONE_PAIR_COST = 25.0
_MULTIRADICAL_ATOM_COST = 2.0
# Tie-breakers, scaled by electronegativity (small enough to never override the above)
_CHARGE_EN_COST = 0.01
_RADICAL_EN_COST = 0.001

_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}
_MAX_BOND_ORDER = 3
_MAX_RADICALS_PER_ATOM = 3

# Default random seed for coordinate embedding, for reproducible geometries
EMBED_SEED = 0xF00D


def from_smiles(smi: str, *, with_coords: bool = False) -> Mol:
    """
    Get RDKit molecule from SMILES string.

    Parameters
    ----------
    smi
        SMILES string.

    with_coords, optional
        If `True`, generate 3D coordinates for the molecule.
        If `False` (default), return a molecule without coordinates.

    Returns
    -------
        RDKit molecule with explicit hydrogens.

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        msg = f"Invalid SMILES: {smi!r}"
        raise ValueError(msg)
    mol = Chem.AddHs(mol)
    if with_coords:
        add_coordinates(mol, in_place=True)
    return mol


def smiles(mol: Mol) -> str:
    """
    Get canonical, non-isomeric SMILES string from Mol.

    Hydrogens are written as they appear in the molecule, so a molecule with
    explicit hydrogens (e.g. from `from_smiles`) gives e.g. ``[H]O[H]`` for water.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        SMILES string.
    """
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def from_inchi(inchi: str, *, with_coords: bool = False) -> Mol:
    """
    Get RDKit molecule from InChI string.

    Parameters
    ----------
    inchi
        InChI string.

    with_coords, optional
        If `True`, generate 3D coordinates for the molecule.
        If `False` (default), return a molecule without coordinates.

    Returns
    -------
        RDKit molecule with explicit hydrogens.

    Raises
    ------
    ValueError
        If the InChI string cannot be parsed.
    """
    mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=False)
    if mol is None:
        msg = f"Invalid InChI: {inchi!r}"
        raise ValueError(msg)
    mol = Chem.AddHs(mol)
    if with_coords:
        add_coordinates(mol, in_place=True)
    return mol


def inchi(mol: Mol) -> str:
    """
    Get standard InChI string from Mol.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        InChI identifier.

    Raises
    ------
    ValueError
        If InChI generation fails.
    """
    ich = Chem.inchi.MolBlockToInchi(Chem.rdmolfiles.MolToMolBlock(mol))
    if not ich:
        msg = f"InChI generation failed for {Chem.MolToSmiles(mol)!r}."
        raise ValueError(msg)
    return ich


def from_xyz_block(xyz_block: str) -> Mol:
    """
    Get RDKit molecule from XYZ.

    Parameters
    ----------
    xyz_block
        Formatted xyz string.

    Returns
    -------
        RDKit molecule.

    Raises
    ------
    ValueError
        If the xyz block cannot be parsed.
    """
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        msg = f"Invalid xyz block: {xyz_block!r}"
        raise ValueError(msg)
    conn_mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    return conn_mol


def xyz_block(mol: Mol) -> str:
    """
    Get formatted xyz block from rdkit molecule.

    Parameters
    ----------
    mol
        `rdkit.Chem.Mol` instance.

    Returns
    -------
    Formatted xyz block.
    """
    return Chem.MolToXYZBlock(mol)


def max_bond_count(symbol: str, *, max_charge: int = 1) -> float:
    """Get the maximum number of bonds to an atom.

    Allows formal charges up to `max_charge` in magnitude, assigning a charged atom
    the valences of its isoelectronic neutral counterpart (as `from_connectivity`
    does), e.g. 4 for N (as in N+), 3 for O (as in O+), and 6 for S.

    Parameters
    ----------
    symbol
        Atomic symbol.
    max_charge, optional
        Maximum magnitude of the formal charge.

    Returns
    -------
        Maximum bond count, or `math.inf` if RDKit places no limit on the valence
        (e.g., for metals).
    """
    periodic_table = Chem.GetPeriodicTable()
    z = element.number(symbol)
    count = 0
    for chg in range(-max_charge, max_charge + 1):
        z_eff = z - chg
        if z_eff <= 0:
            continue
        valences = list(periodic_table.GetValenceList(z_eff))
        if -1 in valences:
            return math.inf
        count = max(count, *valences)
    return count


@dataclass(frozen=True, slots=True)
class _AtomState:
    """Candidate (charge, valence) state of an atom in bond perception."""

    atom: int
    charge: int
    valence: int
    cost: float


def _atom_states(
    idx: int, symbol: str, degree: int, max_charge: int
) -> list[_AtomState]:
    """Enumerate candidate (charge, valence) states of an atom.

    A charged atom is assigned the allowed valences of its isoelectronic neutral
    counterpart (e.g. N+ ~ C, O- ~ F, F- ~ Ne), mirroring RDKit's valence model.
    """
    if element.is_metal(symbol):
        msg = f"Bond perception is not implemented for metals (atom {idx}: {symbol})."
        raise NotImplementedError(msg)

    periodic_table = Chem.GetPeriodicTable()
    z = element.number(symbol)
    en = element.electronegativity(symbol) or 0.0

    states = []
    for chg in range(-max_charge, max_charge + 1):
        z_eff = z - chg
        if z_eff < 0:
            continue
        valences = [0] if z_eff == 0 else list(periodic_table.GetValenceList(z_eff))
        if -1 in valences:
            continue

        cost = _CHARGE_COST * abs(chg) + _CHARGE_EN_COST * chg * en
        states.extend(
            _AtomState(idx, chg, val, cost + _EXPANDED_VALENCE_COST * rank)
            for rank, val in enumerate(valences)
            if val >= degree
        )
        # Closed-shell state with an extra non-bonding pair (e.g. singlet carbene)
        val = valences[0] - 2
        if val >= degree and val not in valences:
            states.append(_AtomState(idx, chg, val, cost + _LONE_PAIR_COST))

    if not states:
        msg = f"Atom {idx} ({symbol}) cannot accommodate {degree} bonds."
        raise GeometryConversionError(msg)

    return states


def _solve_lewis_structure(
    symbols: Sequence[str],
    edges: Sequence[tuple[int, int]],
    *,
    charge: int,
    spin: int,
) -> tuple[list[int], list[int], list[int]]:
    """Solve the bond perception integer program.

    Returns
    -------
        Formal charge and radical electron count per atom, and order per bond.
    """
    natms = len(symbols)
    degrees = np.zeros(natms, dtype=int)
    for i, j in edges:
        degrees[[i, j]] += 1

    max_charge = max(1, abs(charge))
    states = [
        state
        for idx, (symb, deg) in enumerate(zip(symbols, degrees, strict=True))
        for state in _atom_states(idx, symb, int(deg), max_charge)
    ]

    # Variables: state selectors | bond orders | 1st radical | additional radicals
    nst, nbd = len(states), len(edges)
    ibd, irad1, irad2 = nst, nst + nbd, nst + nbd + natms
    nvar = irad2 + natms

    ens = np.array([element.electronegativity(s) or 0.0 for s in symbols])
    cost = np.zeros(nvar)
    cost[:nst] = [s.cost for s in states]
    cost[irad1:irad2] = _RADICAL_EN_COST * ens
    cost[irad2:] = _MULTIRADICAL_ATOM_COST + _RADICAL_EN_COST * ens

    lower = np.zeros(nvar)
    upper = np.ones(nvar)
    lower[ibd:irad1] = 1
    upper[ibd:irad1] = _MAX_BOND_ORDER
    upper[irad2:] = _MAX_RADICALS_PER_ATOM - 1

    # Rows: one state per atom | valence saturation per atom | total charge | spin
    amat = np.zeros((2 * natms + 2, nvar))
    rhs = np.zeros(2 * natms + 2)
    rhs[:natms] = 1
    rhs[-2:] = charge, spin
    for k, state in enumerate(states):
        amat[state.atom, k] = 1
        amat[natms + state.atom, k] = -state.valence
        amat[-2, k] = state.charge
    for k, (i, j) in enumerate(edges):
        amat[natms + i, ibd + k] = amat[natms + j, ibd + k] = 1
    for i in range(natms):
        amat[natms + i, [irad1 + i, irad2 + i]] = 1
        amat[-1, [irad1 + i, irad2 + i]] = 1

    res = milp(
        cost,
        constraints=LinearConstraint(amat, rhs, rhs),
        integrality=np.ones(nvar),
        bounds=Bounds(lower, upper),
    )
    if res.x is None:
        msg = (
            f"No Lewis structure found for charge {charge} and spin {spin}: "
            f"{res.message}"
        )
        raise GeometryConversionError(msg)

    sol = np.rint(res.x).astype(int)
    charges = [0] * natms
    for k, state in enumerate(states):
        if sol[k]:
            charges[state.atom] = state.charge
    radicals = (sol[irad1:irad2] + sol[irad2:]).tolist()
    orders = sol[ibd:irad1].tolist()
    return charges, radicals, orders


def from_connectivity(
    symbols: Sequence[str],
    bonds: Collection[tuple[int, int]],
    *,
    charge: int = 0,
    spin: int = 0,
    coords: ArrayLike | None = None,
) -> Mol:
    """
    Get RDKit molecule from atoms and connectivity, perceiving the Lewis structure.

    Bond orders, formal charges, and radical electrons are determined by solving an
    integer linear program over the bonded atoms:

    - Each atom takes one (charge, valence) state from RDKit's valence model, where a
      charged atom uses the valences of its isoelectronic neutral counterpart.
    - Each bond takes an order of 1, 2, or 3.
    - Each atom's valence is saturated by its bond orders plus radical electrons.
    - Formal charges sum to `charge` and radical electrons sum to `spin`.

    The objective penalizes, in decreasing order of severity, closed-shell lone pairs
    beyond the default valence (singlet carbenes), formal charges, valence expansion,
    and multiple radical electrons on one atom. Ties are broken by placing negative
    charge on more electronegative atoms, positive charge and radicals on less
    electronegative ones.

    Parameters
    ----------
    symbols
        Atomic symbols.
    bonds
        Pairs of bonded atom indices.
    charge, optional
        Total molecular charge.
    spin, optional
        Number of unpaired electrons.
    coords, optional
        Atomic coordinates as an (N, 3) array, added as a conformer if given.

    Returns
    -------
        Sanitized RDKit molecule with explicit hydrogens.

    Raises
    ------
    NotImplementedError
        If the molecule contains a metal.
    GeometryConversionError
        If no Lewis structure is consistent with the connectivity, charge, and spin.
    """
    natms = len(symbols)
    edges = sorted({(min(i, j), max(i, j)) for i, j in map(tuple, bonds)})
    if any(i == j or not 0 <= i < j < natms for i, j in edges):
        msg = f"Invalid bonds for {natms} atoms: {edges}"
        raise GeometryConversionError(msg)

    nelec = sum(map(element.number, symbols)) - charge
    if spin < 0 or spin > nelec or (nelec - spin) % 2:
        msg = f"Spin {spin} is inconsistent with {nelec} electrons (charge {charge})."
        raise GeometryConversionError(msg)

    charges, radicals, orders = _solve_lewis_structure(
        symbols, edges, charge=charge, spin=spin
    )

    rwmol = Chem.RWMol()
    for symb, chg, nrad in zip(symbols, charges, radicals, strict=True):
        atom = Chem.Atom(element.number(symb))
        atom.SetFormalCharge(chg)
        atom.SetNumRadicalElectrons(int(nrad))
        atom.SetNoImplicit(True)  # noqa: FBT003
        rwmol.AddAtom(atom)
    for (i, j), order in zip(edges, orders, strict=True):
        rwmol.AddBond(i, j, _BOND_TYPES[order])

    mol = rwmol.GetMol()
    if coords is not None:
        set_coordinates(mol, np.asarray(coords, dtype=np.float64), in_place=True)

    try:
        # Radicals are assigned explicitly; RDKit would otherwise turn closed-shell
        # lone pairs (e.g., singlet carbenes) into radicals.
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
        )
    except Chem.MolSanitizeException as err:
        msg = f"Perceived Lewis structure failed sanitization: {err}"
        raise GeometryConversionError(msg) from err

    return mol


def symbols(mol: Mol) -> list[str]:
    """
    Get atomic symbols.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        List of atomic symbols.
    """
    return [a.GetSymbol() for a in mol.GetAtoms()]


def coordinates(mol: Mol) -> FloatArray:
    """
    Get atom coordinates.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        Atomic coordinates as (N, 3) numpy array.

    Raises
    ------
    GeometryConversionError
        If the molecule has no coordinates.
    """
    if not has_coordinates(mol):
        msg = "Molecule has no coordinates. Did you forget to add them?"
        raise GeometryConversionError(msg)

    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def charge(mol: Mol) -> int:
    """
    Get molecular charge.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        Molecular charge as an integer.
    """
    return Chem.GetFormalCharge(mol)


def spin(mol: Mol) -> int:
    """
    Get molecular spin (number of unpaired electrons).

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        Number of unpaired electrons as an integer.
    """
    return Descriptors.NumRadicalElectrons(mol)


# Boolean properties
def has_coordinates(mol: Mol) -> bool:
    """
    Check if coordinates have been added.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        `True` if the molecule has coordinates, False otherwise.
    """
    return bool(mol.GetNumConformers())


# Transformations
def set_coordinates(mol: Mol, coords: FloatArray, *, in_place: bool = False) -> Mol:
    """
    Set atom coordinates, replacing any existing conformer.

    Parameters
    ----------
    mol
        RDKit molecule object.
    coords
        Atomic coordinates as an (N, 3) array.
    in_place, optional
        If `True`, modify the molecule in place.
        If `False` (default), return a new molecule.

    Returns
    -------
        RDKit molecule object with the given coordinates.

    Raises
    ------
    ValueError
        If `coords` is not of shape (N, 3) for N atoms.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (mol.GetNumAtoms(), 3):
        msg = (
            f"Expected coordinates of shape ({mol.GetNumAtoms()}, 3), "
            f"got {coords.shape}."
        )
        raise ValueError(msg)

    mol = mol if in_place else Mol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def add_coordinates(
    mol: Mol, *, seed: int | None = EMBED_SEED, in_place: bool = False
) -> Mol:
    """
    Add coordinates, if missing.

    Parameters
    ----------
    mol
        RDKit molecule object.
    seed, optional
        Random seed for embedding, for reproducible coordinates.
        If `None`, use a random seed.
    in_place, optional
        If `True`, modify the molecule in place.
        If `False` (default), return a new molecule.

    Returns
    -------
        RDKit molecule object with coordinates.
        (Unmodified if it already had coordinates.)

    Raises
    ------
    GeometryConversionError
        If coordinates cannot be embedded.
    """
    if has_coordinates(mol):
        return mol

    mol = mol if in_place else Mol(mol)
    seed = -1 if seed is None else seed
    if EmbedMolecule(mol, randomSeed=seed) == -1 and (
        EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) == -1
    ):
        msg = f"Failed to embed coordinates for {Chem.MolToSmiles(mol)!r}."
        raise GeometryConversionError(msg)
    return mol


def add_atom_numbers(
    mol: Mol, to_number: Mapping[int, int], *, in_place: bool = False
) -> Mol:
    """Add atom numbers.

    Parameters
    ----------
    mol
        RDKit molecule object.
    to_number
        Mapping from atom index to atom number.
    in_place, optional
        If `True`, modify the molecule in place.
        If `False` (default), return a new molecule.

    Returns
    -------
        RDKit molecule object with "atomLabel" property set to f"{symbol}{number}"
    """
    mol = mol if in_place else Mol(mol)
    for atom in mol.GetAtoms():
        number = to_number[atom.GetIdx()]
        symbol = atom.GetSymbol()
        atom.SetProp("atomLabel", f"{symbol}{number}")

    return mol


def canonical_ranks(mol: Mol, *, break_ties: bool = True) -> list[int]:
    """Return the canonical ranking.

    Parameters
    ----------
    mol
        RDKit molecule object.
    break_ties
        If `True`, force breaking of ranked ties.

    Returns
    -------
        List of canonical ranks.
    """
    ranks = Chem.CanonicalRankAtoms(mol=mol, breakTies=break_ties)
    return cast("list[int]", ranks)


def assign_stereochemistry(mol: Mol, *, in_place: bool = False) -> Mol:
    """Assign stereochemistry from 3D coordinates.

    Parameters
    ----------
    mol
        RDKit molecule object.
    in_place
        If `True`, modify the molecule in place.
        If `False` (default), return a new molecule.

    Returns
    -------
        RDKit molecule object with stereochemistry tags.
    """
    mol = mol if in_place else Mol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def chiral_centers(mol: Mol) -> list[tuple[int, str]]:
    """Return chiral center indices with R/S labels.

    Parameters
    ----------
    mol
        RDKit molecule object.

    Returns
    -------
        List of (index, label) for all chiral centers.
    """
    mol = assign_stereochemistry(mol)
    return Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)
