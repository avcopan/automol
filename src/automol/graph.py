"""Core functions."""

import copy
import operator as op
from enum import StrEnum

import networkx as nx
from networkx.algorithms.isomorphism import ISMAGS
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


# Comparisons
def is_isomorphic(G1: nx.Graph, G2: nx.Graph, *, bond_orders: bool = False) -> bool:  # noqa: N803
    """Check if two graphs are isomorphic."""
    if not bond_orders:
        G1 = remove_bond_orders(G1)  # noqa: N806
        G2 = remove_bond_orders(G2)  # noqa: N806
    return nx.is_isomorphic(G1, G2, node_match=op.eq, edge_match=op.eq)


def mcs_mappings(G1: nx.Graph, G2: nx.Graph) -> list[dict[int, int]]:  # noqa: N803
    """Find the maximum common subgraph between two graphs."""
    ismags = ISMAGS(G1, G2, node_match=op.eq, edge_match=op.eq)
    return list(ismags.largest_common_subgraph())


def reaction_mappings_mcs_recursive(
    G1: nx.Graph,  # noqa: N803
    G2: nx.Graph,  # noqa: N803
) -> list[dict[int, int]]:
    """Find the maximum common subgraph between two graphs using a recursive approach.

    Does not work well. Need to implement algorithm from autoDE.
    """
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return [{}]

    mappings = []

    for mapping in mcs_mappings(G1, G2):
        nodes1 = set(G1.nodes()) - set(mapping.keys())
        nodes2 = set(G2.nodes()) - set(mapping.values())

        sub_mappings = reaction_mappings_mcs_recursive(
            G1.subgraph(nodes1), G2.subgraph(nodes2)
        )

        mappings.extend({**mapping, **sub_mapping} for sub_mapping in sub_mappings)

    return mappings


# Transition state graphs
def transition_state_graphs(G1: nx.Graph, G2: nx.Graph) -> list[nx.Graph]:  # noqa: N803
    """Construct a transition graph between two graphs.

    Does not work well. Need to implement algorithm from autoDE.
    """
    TGs = []  # noqa: N806

    # Note: Using reverse mappings to map the product onto the reactant
    for mapping in reaction_mappings_mcs_recursive(G2, G1):
        G2_ = nx.relabel_nodes(G2, mapping)  # noqa: N806

        formed_bonds = set(G2_.edges()) - set(G1.edges())
        broken_bonds = set(G1.edges()) - set(G2_.edges())

        formed_bond_model = Bond(change=Change.FORMED, order=1)
        broken_bond_model = Bond(change=Change.BROKEN, order=1)

        TG = remove_bond_orders(G1)  # noqa: N806
        TG.add_edges_from(formed_bonds, **formed_bond_model.model_dump())
        TG.add_edges_from(broken_bonds, **broken_bond_model.model_dump())

        validate(TG)

        RG = reactants_graph(TG)  # noqa: N806
        PG = products_graph(TG)  # noqa: N806

        if is_isomorphic(RG, G1) and is_isomorphic(PG, G2):
            TGs.append(TG)

    # Filter out redundant transition graphs
    unique_transition_graphs = []
    for TG in TGs:  # noqa: N806
        if not any(is_isomorphic(TG, G) for G in unique_transition_graphs):
            unique_transition_graphs.append(TG)

    return unique_transition_graphs


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
