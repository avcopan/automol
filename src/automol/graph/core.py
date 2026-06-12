"""Core molecular graph functions.

Uses NetworkX for graph representation, with Atom and Bond data validation.

Excludes bond order information by design.
"""

import copy
from collections.abc import Collection, Iterator, Sequence
from typing import Any, TypeVar

import networkx as nx
import numpy as np
from automatics import element, rd
from networkx.algorithms.isomorphism import GraphMatcher
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import Mol, RWMol

BondKey = tuple[int, int]


class _CustomBaseModelMeta(ModelMetaclass):
    def __getattr__(self, item: str):  # noqa: ANN204
        try:
            super().__getattr__(item)  # ty:ignore[unresolved-attribute]
        except AttributeError:
            if item in self.__dict__.get("__pydantic_fields__", ()):
                return item
            raise


class CustomBaseModel(BaseModel, metaclass=_CustomBaseModelMeta):
    """A custom base model that allows accessing field names as class attributes."""


class Atom(CustomBaseModel):
    """Represents an atom in a molecule."""

    symbol: str

    def to_rdkit_atom(self) -> rdchem.Atom:
        """Convert to an RDKit Atom."""
        rd_atom = rdchem.Atom(self.symbol)
        rd_atom.SetNoImplicit(True)  # noqa: FBT003
        return rd_atom


class Bond(CustomBaseModel):
    """Represents a bond between two atoms in a molecule."""

    def to_rdkit_bond_type(self) -> rdchem.BondType:
        """Convert to an RDKit Bond Type."""
        return rdchem.BondType.SINGLE


AtomT = TypeVar("AtomT", bound=Atom)
BondT = TypeVar("BondT", bound=Bond)


class Graph[AtomT: Atom, BondT: Bond](nx.Graph):
    """Generic chemical graph."""

    atom_type: type[AtomT]
    bond_type: type[BondT]

    def validate(self) -> None:
        """Validate against atom and bond classes."""
        for *_, data in self.nodes(data=True):
            self.atom_type.model_validate(data)

        for *_, data in self.edges(data=True):
            self.bond_type.model_validate(data)


class MolGraph(Graph[Atom, Bond]):
    """Molecular graph."""

    atom_type = Atom
    bond_type = Bond


# Properties
def atom_keys[AtomT: Atom, BondT: Bond](gra: Graph[AtomT, BondT]) -> list[int]:
    """Get list of atom keys."""
    return list(gra.nodes())


def symbols[AtomT: Atom, BondT: Bond](
    gra: Graph[AtomT, BondT], keys: Sequence[int] | None = None
) -> list[str]:
    """Get symbols of atoms."""
    keys = atom_keys(gra) if keys is None else keys
    return [gra.nodes[key][Atom.symbol] for key in keys]


def element_bonding_capacities[AtomT: Atom, BondT: Bond](
    gra: Graph[AtomT, BondT], keys: Sequence[int] | None = None
) -> list[int]:
    """Get element valences of atoms."""
    symbs = symbols(gra, keys)
    return [element.bonding_capacity(symb) for symb in symbs]


def degrees(gra: MolGraph, keys: Sequence[int] | None = None) -> list[int]:
    """Get degrees of atoms."""
    keys = atom_keys(gra) if keys is None else keys
    return [gra.degree[key] for key in keys]


def open_valences(gra: MolGraph, keys: Sequence[int] | None = None) -> list[int]:
    """Get open valences of atoms.

    This is the capacity minus the degree, i.e. the number of additional bonds
    that could be formed.
    """
    caps = element_bonding_capacities(gra, keys)
    degs = degrees(gra, keys)
    return np.subtract(caps, degs).tolist()


# Transformations
def remove_bonds[AtomT: Atom, BondT: Bond](
    gra: Graph[AtomT, BondT],
    bonds: Collection[tuple[int, int]],
    *,
    in_place: bool = False,
) -> Graph[AtomT, BondT]:
    """Return a copy of the graph with specified bonds removed."""
    gra = gra if in_place else copy.deepcopy(gra)
    gra.remove_edges_from(bonds)
    return gra


# Algorithms
def isomorphisms[AtomT: Atom, BondT: Bond](
    gra1: Graph[AtomT, BondT], gra2: Graph[AtomT, BondT]
) -> Iterator[dict[int, int]]:
    """Check if two graphs are isomorphic."""
    return graph_matcher(gra1, gra2).isomorphisms_iter()


def isomorphism[AtomT: Atom, BondT: Bond](
    gra1: Graph[AtomT, BondT], gra2: Graph[AtomT, BondT]
) -> dict[int, int] | None:
    """Check if two graphs are isomorphic.

    Does not consider bond orders.
    """
    return next(isomorphisms(gra1, gra2), None)


# Comparisons
def is_isomorphic[AtomT: Atom, BondT: Bond](
    gra1: Graph[AtomT, BondT], gra2: Graph[AtomT, BondT]
) -> bool:
    """Check if two graphs are isomorphic."""
    return graph_matcher(gra1, gra2).is_isomorphic()


def graph_matcher[AtomT: Atom, BondT: Bond](
    gra1: Graph[AtomT, BondT], gra2: Graph[AtomT, BondT]
) -> GraphMatcher:
    """Check if two graphs are isomorphic."""
    atom_fields = gra1.atom_type.model_fields.keys()
    bond_fields = gra1.bond_type.model_fields.keys()

    def atom_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
        return all(n1[field] == n2[field] for field in atom_fields)

    def bond_match(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        return all(e1[field] == e2[field] for field in bond_fields)

    return GraphMatcher(gra1, gra2, node_match=atom_match, edge_match=bond_match)


# Conversions from other types
def from_smiles(smi: str) -> MolGraph:
    """
    Instantiate Graph from SMILES string.

    Parameters
    ----------
    smi
        SMILES formatted string.

    Returns
    -------
    MolGraph
        MolGraph.
    """
    mol = rd.mol.from_smiles(smi, with_coords=False)
    return from_rdkit_mol(mol)


def from_inchi(chi: str) -> MolGraph:
    """
    Instantiate Graph from InChI string.

    Parameters
    ----------
    chi
        InChI string.

    Returns
    -------
    MolGraph
        MolGraph.
    """
    mol = rd.mol.from_inchi(chi, with_coords=False)
    return from_rdkit_mol(mol)


def from_rdkit_mol(mol: Mol) -> MolGraph:
    """
    Instantiate Graph from RKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
    MolGraph
        MolGraph.
    """
    gra = MolGraph(atom_type=Atom, bond_type=Bond)

    for rd_atom in mol.GetAtoms():
        atom = Atom(symbol=rd_atom.GetSymbol())
        gra.add_node(rd_atom.GetIdx(), **atom.model_dump())

    for rd_bond in mol.GetBonds():
        bond = Bond()
        gra.add_edge(
            rd_bond.GetBeginAtomIdx(), rd_bond.GetEndAtomIdx(), **bond.model_dump()
        )
    gra.validate()
    return gra


def inchi(gra: MolGraph) -> str:
    """
    Provide InChI string from Graph.

    Parameters
    ----------
    gra
        MolGraph object.

    Returns
    -------
    str
        InChI string.
    """
    mol = rdkit_mol(gra)
    return rd.mol.inchi(mol)


def rdkit_mol[AtomT: Atom, BondT: Bond](
    gra: Graph[AtomT, BondT],
    *,
    label: bool = False,
) -> Mol:
    """Convert a graph back to an RDKit molecule."""
    mol, to_key = rdkit_mol_with_index_map(gra)
    if label:
        mol = rd.mol.add_atom_numbers(mol, to_number=to_key)
    return mol


def rdkit_mol_with_index_map[AtomT: Atom, BondT: Bond](
    gra: Graph[AtomT, BondT],
) -> tuple[Mol, dict[int, int]]:
    """Convert a graph back to an RDKit molecule."""
    rw_mol = RWMol()
    to_idx: dict[int, int] = {}

    for key in sorted(gra.nodes()):
        atom = gra.atom_type.model_validate(gra.nodes[key])
        idx = rw_mol.AddAtom(atom.to_rdkit_atom())
        to_idx[key] = idx

    for key1, key2 in gra.edges():
        bond = gra.bond_type.model_validate(gra.edges[key1, key2])
        rw_mol.AddBond(to_idx[key1], to_idx[key2], order=bond.to_rdkit_bond_type())

    to_key = dict(map(reversed, to_idx.items()))
    return rw_mol.GetMol(), to_key
