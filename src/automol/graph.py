"""Core functions."""

import copy
import itertools
import operator as op
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator
from enum import StrEnum

import networkx as nx
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import Mol, RWMol

from . import rd


class Change(StrEnum):
    """Changes."""

    FORMED = "formed"
    BROKEN = "broken"
    FLEETING = "fleeting"


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


class Bond(CustomBaseModel):
    """Represents a bond between two atoms in a molecule."""

    change: Change | None = None
    order: int


def validate(G: nx.Graph) -> nx.Graph:  # noqa: N803
    """Validate the graph structure."""
    for key, data in G.nodes(data=True):
        if not Atom.model_validate(data):
            msg = f"Node {key} does not have a valid Atom instance."
            raise ValueError(msg)
    for key1, key2, data in G.edges(data=True):
        if not Bond.model_validate(data):
            msg = f"Edge ({key1}, {key2}) does not have a valid Bond instance."
            raise ValueError(msg)

    return G


def from_smiles(smi: str) -> nx.Graph:
    """
    Instantiate Graph from SMILES string.

    Parameters
    ----------
    smi
        SMILES formatted string.

    Returns
    -------
    graph
        Graph.
    """
    mol = rd.mol.from_smiles(smi, with_coords=False)
    return from_rdkit_mol(mol)


def from_inchi(chi: str) -> nx.Graph:
    """
    Instantiate Graph from InChI string.

    Parameters
    ----------
    chi
        InChI string.

    Returns
    -------
    graph
        Graph.
    """
    mol = rd.mol.from_inchi(chi, with_coords=False)
    return from_rdkit_mol(mol)


def from_rdkit_mol(mol: Mol) -> nx.Graph:
    """
    Instantiate Graph from RKit molecule.

    Parameters
    ----------
    mol
        RDKit molecule.

    Returns
    -------
    graph
        Graph.
    """
    graph = nx.Graph()

    for mol_atom in mol.GetAtoms():
        atom = Atom(symbol=mol_atom.GetSymbol())
        graph.add_node(mol_atom.GetIdx(), **atom.model_dump())

    for mol_bond in mol.GetBonds():
        bond = Bond(order=mol_bond.GetBondTypeAsDouble())
        graph.add_edge(
            mol_bond.GetBeginAtomIdx(), mol_bond.GetEndAtomIdx(), **bond.model_dump()
        )

    return validate(graph)


def inchi(G: nx.Graph) -> str:  # noqa: N803
    """
    Provide InChI string from Graph.

    Parameters
    ----------
    G
        Graph object.

    Returns
    -------
    xyz
        Formatted xyz block.
    """
    mol = rdkit_mol(G)
    return rd.mol.inchi(mol)


def rdkit_mol(G: nx.Graph, *, label: bool = False) -> Mol:  # noqa: N803
    """Convert a graph back to an RDKit molecule."""
    mol, to_key = rdkit_mol_with_index_map(G)
    if label:
        mol = rd.mol.add_atom_numbers(mol, to_number=to_key)
    return mol


# Transformations
def remove_bond_orders(G: nx.Graph, *, in_place: bool = False) -> nx.Graph:  # noqa: N803
    """Return a copy of the graph without bond order information."""
    G = G if in_place else copy.deepcopy(G)  # noqa: N806
    for key1, key2 in G.edges():
        G.edges[key1, key2][Bond.order] = 1
    return G


def remove_bonds(
    G: nx.Graph,  # noqa: N803
    bonds: Collection[tuple[int, int]],
    *,
    in_place: bool = False,
) -> nx.Graph:
    """Return a copy of the graph with specified bonds removed."""
    G = G if in_place else copy.deepcopy(G)  # noqa: N806
    G.remove_edges_from(bonds)
    return G


# Comparisons
def isomorphisms(G1: nx.Graph, G2: nx.Graph) -> list[dict[int, int]]:  # noqa: N803
    """Check if two graphs are isomorphic.

    Does not consider bond orders.
    """
    return list(nx.vf2pp_all_isomorphisms(G1, G2, node_label=Atom.symbol))


def isomorphism(G1: nx.Graph, G2: nx.Graph) -> dict[int, int] | None:  # noqa: N803
    """Check if two graphs are isomorphic.

    Does not consider bond orders.
    """
    return nx.vf2pp_isomorphism(G1, G2, node_label=Atom.symbol)


def is_isomorphic(G1: nx.Graph, G2: nx.Graph, *, bond_orders: bool = False) -> bool:  # noqa: N803
    """Check if two graphs are isomorphic."""
    if not bond_orders:
        G1 = remove_bond_orders(G1)  # noqa: N806
        G2 = remove_bond_orders(G2)  # noqa: N806
    return nx.is_isomorphic(G1, G2, node_match=op.eq, edge_match=op.eq)


# Transition state graphs
BondKey = tuple[int, int]
BondType = tuple[str, str]
FORMED_BOND = Bond(change=Change.FORMED, order=1)
BROKEN_BOND = Bond(change=Change.BROKEN, order=1)


def transition_state_graph(
    G: nx.Graph,  # noqa: N803
    bond_changes: dict[BondKey, Change],
) -> nx.Graph:
    """Construct a transition graph from a graph and bond changes."""
    TG = remove_bond_orders(G)  # noqa: N806
    formed_bonds = {k for k, c in bond_changes.items() if c == Change.FORMED}
    broken_bonds = {k for k, c in bond_changes.items() if c == Change.BROKEN}
    TG.add_edges_from(formed_bonds, **FORMED_BOND.model_dump())
    TG.add_edges_from(broken_bonds, **BROKEN_BOND.model_dump())
    return validate(TG)


def formed_and_broken_bonds(
    G: nx.Graph,  # noqa: N803
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Extract the formed and broken bonds from a transition graph."""
    change = nx.get_edge_attributes(G, Bond.change)
    formed = {k for k, v in change.items() if v == Change.FORMED}
    broken = {k for k, v in change.items() if v == Change.BROKEN}
    return formed, broken


def formed_bonds(G: nx.Graph) -> set[tuple[int, int]]:  # noqa: N803
    """Extract the formed bonds from a transition graph."""
    formed, _ = formed_and_broken_bonds(G)
    return formed


def broken_bonds(G: nx.Graph) -> set[tuple[int, int]]:  # noqa: N803
    """Extract the broken bonds from a transition graph."""
    _, broken = formed_and_broken_bonds(G)
    return broken


def reverse(G: nx.Graph, *, in_place: bool = False) -> nx.Graph:  # noqa: N803
    """Reverse the direction of a transition graph."""
    G = G if in_place else copy.deepcopy(G)  # noqa: N806
    form_bnds, brok_bnds = formed_and_broken_bonds(G)
    new_bnds = {}
    new_bnds.update(dict.fromkeys(brok_bnds, Change.FORMED))
    new_bnds.update(dict.fromkeys(form_bnds, Change.BROKEN))
    nx.set_edge_attributes(G, new_bnds, Bond.change)
    return G


def reactants_graph(G: nx.Graph) -> nx.Graph:  # noqa: N803
    """Extract the reactant graph from a transition graph."""
    form_bnds, _ = formed_and_broken_bonds(G)
    rct_graph = copy.deepcopy(G)
    nx.set_edge_attributes(rct_graph, None, Bond.change)
    rct_graph.remove_edges_from(form_bnds)
    return rct_graph


def products_graph(G: nx.Graph) -> nx.Graph:  # noqa: N803
    """Extract the reactant graph from a transition graph."""
    return reactants_graph(reverse(G))


# Reaction mapping
def transition_state_graphs(
    G1: nx.Graph,  # noqa: N803
    G2: nx.Graph,  # noqa: N803
) -> list[nx.Graph]:
    """Fewest-bonds-first constructive count vector mappings."""
    TGs, _ = transition_state_graphs_with_mappings(G1, G2)  # noqa: N806
    return TGs


def transition_state_graphs_with_mappings(
    G1: nx.Graph,  # noqa: N803
    G2: nx.Graph,  # noqa: N803
) -> tuple[list[nx.Graph], list[dict[int, int]]]:
    """Fewest-bonds-first constructive count vector mappings.

    Note: The mappings are from products to reactants!
    """
    bond_types1 = _bond_types(G1)
    bond_types2 = _bond_types(G2)
    counter1 = Counter(bond_types1.values())
    counter2 = Counter(bond_types2.values())
    diff_counter = copy.deepcopy(counter2)
    diff_counter.subtract(counter1)

    form_counts = {k: v for k, v in diff_counter.items() if v > 0}
    break_counts = {k: abs(v) for k, v in diff_counter.items() if v < 0}

    G1 = remove_bond_orders(G1)  # noqa: N806
    G2 = remove_bond_orders(G2)  # noqa: N806

    TGs = []  # noqa: N806
    mappings = []
    for break_bonds1 in _iterate_bond_sets(G1, break_counts):
        for break_bonds2 in _iterate_bond_sets(G2, form_counts):
            G1_ = remove_bonds(G1, break_bonds1)  # noqa: N806
            G2_ = remove_bonds(G2, break_bonds2)  # noqa: N806

            mappings = isomorphisms(G2_, G1_)
            for mapping in mappings:
                break_bonds = break_bonds1
                form_bonds = {tuple(sorted(map(mapping.get, b))) for b in break_bonds2}
                bond_changes = {
                    **dict.fromkeys(break_bonds, Change.BROKEN),
                    **dict.fromkeys(form_bonds, Change.FORMED),
                }
                TG = transition_state_graph(G1, bond_changes)  # noqa: N806
                RG = reactants_graph(TG)  # noqa: N806
                PG = products_graph(TG)  # noqa: N806

                # Continue if reactant does not match
                if not is_isomorphic(RG, G1):
                    continue

                # Continue if product does not match
                if not is_isomorphic(PG, G2):
                    continue

                # Continue if not unique
                if any(is_isomorphic(TG, T) for T in TGs):
                    continue

                TGs.append(TG)
                mappings.append(mapping)

    return TGs, mappings


def _bond_types(G: nx.Graph) -> dict[BondKey, BondType]:  # noqa: N803
    """Extract the bond types from a transition graph."""
    return {
        (key1, key2): tuple(
            sorted([G.nodes[key1][Atom.symbol], G.nodes[key2][Atom.symbol]])
        )
        for key1, key2 in G.edges()
    }


def _bonds_by_type(G: nx.Graph) -> dict[BondType, set[BondKey]]:  # noqa: N803
    """Group bonds by their types."""
    bond_types = _bond_types(G)
    bonds_by_type = defaultdict(set)
    for bond_key, bond_type in bond_types.items():
        bonds_by_type[bond_type].add(bond_key)
    return dict(bonds_by_type)


def _iterate_bond_sets(
    G: nx.Graph,  # noqa: N803
    counts: dict[BondType, int],
) -> Iterator[tuple[BondKey, ...]]:
    """Iterate over all combinations of bonds to form or break."""
    bonds_by_type = _bonds_by_type(G)
    combo_iters = [
        itertools.combinations(bonds_by_type[bond_type], count)
        for bond_type, count in counts.items()
    ]

    for combos in itertools.product(*combo_iters):
        yield tuple(itertools.chain.from_iterable(combos))


# Helpers
def rdkit_mol_with_index_map(G: nx.Graph) -> tuple[Mol, dict[int, int]]:  # noqa: N803
    """Convert a graph back to an RDKit molecule."""
    rw_mol = RWMol()
    to_idx: dict[int, int] = {}

    for key in sorted(G.nodes()):
        atom = Atom(**G.nodes[key])
        rd_atom = rdchem.Atom(atom.symbol)
        rd_atom.SetNoImplicit(True)  # noqa: FBT003
        idx = rw_mol.AddAtom(rd_atom)
        to_idx[key] = idx

    for key1, key2 in G.edges():
        bond = Bond(**G.edges[key1, key2])
        if bond.change is not None:
            rw_mol.AddBond(to_idx[key1], to_idx[key2], rdchem.BondType.HYDROGEN)
        else:
            rw_mol.AddBond(to_idx[key1], to_idx[key2], rdchem.BondType(bond.order))

    to_key = dict(map(reversed, to_idx.items()))
    return rw_mol.GetMol(), to_key
