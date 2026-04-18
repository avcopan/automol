"""Core molecular graph functions.

Uses NetworkX for graph representation, with Atom and Bond data validation.

Excludes bond order information by design.
"""

import copy
from collections.abc import Collection
from typing import Any, TypeVar

import networkx as nx
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import Mol, RWMol

from .. import rd


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
    """Graph representation with Atom and Bond data validation."""

    atom_type: type[AtomT]
    bond_type: type[BondT]

    def __init__(
        self,
        *,
        atom_type: type[AtomT],
        bond_type: type[BondT],
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        self.atom_type = atom_type
        self.bond_type = bond_type

    def validate(self) -> None:
        """Validate against atom and bond classes."""
        for *_, data in self.nodes(data=True):
            self.atom_type.model_validate(data)

        for *_, data in self.edges(data=True):
            self.bond_type.model_validate(data)


# Conversions from other types
def from_smiles(smi: str) -> Graph[Atom, Bond]:
    """
    Instantiate Graph from SMILES string.

    Parameters
    ----------
    smi
        SMILES formatted string.

    Returns
    -------
    Graph
        Graph.
    """
    mol = rd.mol.from_smiles(smi, with_coords=False)
    return from_rdkit_mol(mol)


def from_inchi(chi: str) -> Graph[Atom, Bond]:
    """
    Instantiate Graph from InChI string.

    Parameters
    ----------
    chi
        InChI string.

    Returns
    -------
    Graph
        Graph.
    """
    mol = rd.mol.from_inchi(chi, with_coords=False)
    return from_rdkit_mol(mol)


def from_rdkit_mol(mol: Mol) -> Graph[Atom, Bond]:
    """
    Instantiate Graph from RKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
    Graph
        Graph.
    """
    gra = Graph[Atom, Bond](atom_type=Atom, bond_type=Bond)

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


# Conversions to other types
def inchi(gra: Graph[Atom, Bond]) -> str:
    """
    Provide InChI string from Graph.

    Parameters
    ----------
    gra
        Graph object.

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


# Transformations
def remove_bonds[AtomT: Atom, BondT: Bond](
    gra: Graph[AtomT, BondT],
    bonds: Collection[tuple[int, int]],
    *,
    in_place: bool = False,
) -> Graph:
    """Return a copy of the graph with specified bonds removed."""
    gra = gra if in_place else copy.deepcopy(gra)
    gra.remove_edges_from(bonds)
    return gra


# Algorithms
def isomorphisms(
    gra1: Graph[Atom, Bond], gra2: Graph[Atom, Bond]
) -> list[dict[int, int]]:
    """Check if two graphs are isomorphic."""
    return list(nx.vf2pp_all_isomorphisms(gra1, gra2, node_label=Atom.symbol))


def isomorphism(
    gra1: Graph[Atom, Bond], gra2: Graph[Atom, Bond]
) -> dict[int, int] | None:
    """Check if two graphs are isomorphic.

    Does not consider bond orders.
    """
    return nx.vf2pp_isomorphism(gra1, gra2, node_label=Atom.symbol)


# Comparisons
def is_isomorphic(gra1: Graph[Atom, Bond], gra2: Graph[Atom, Bond]) -> bool:
    """Check if two graphs are isomorphic."""
    return nx.vf2pp_is_isomorphic(gra1, gra2, node_label=Atom.symbol)
