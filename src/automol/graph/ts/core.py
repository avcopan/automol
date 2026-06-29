"""Core molecular graph functions.

Uses NetworkX for graph representation, with Atom and Bond data validation.

Does not include bond order information.
"""

from enum import StrEnum

import networkx as nx
from rdkit.Chem import rdchem

from ..core import Atom, Bond, BondKey, Graph, MolGraph


class Change(StrEnum):
    """Changes."""

    FORMED = "formed"
    BROKEN = "broken"
    FLEETING = "fleeting"


class TransBond(Bond):
    """Represents a bond between two atoms in a molecule."""

    change: Change | None

    def to_rdkit_bond_type(self) -> rdchem.BondType:
        """Convert to an RDKit Bond Type."""
        if self.change is not None:
            return rdchem.BondType.HYDROGEN
        return rdchem.BondType.SINGLE


# Transition state graphs


class TransGraph(Graph[Atom, TransBond]):
    """Molecular graph."""

    atom_type = Atom
    bond_type = TransBond


FORMED_BOND = TransBond(change=Change.FORMED)
BROKEN_BOND = TransBond(change=Change.BROKEN)


def from_bond_changes(gra: MolGraph, bond_changes: dict[BondKey, Change]) -> TransGraph:
    """Construct a transition graph from a graph and bond changes."""
    ts_gra = TransGraph()
    ts_gra.add_nodes_from(gra.nodes(data=True))
    ts_gra.add_edges_from(gra.edges(data=True), change=None)
    formed_bonds = {k for k, c in bond_changes.items() if c == Change.FORMED}
    broken_bonds = {k for k, c in bond_changes.items() if c == Change.BROKEN}
    ts_gra.add_edges_from(formed_bonds, change=Change.FORMED, distance=None)
    ts_gra.add_edges_from(broken_bonds, change=Change.BROKEN)
    ts_gra.validate()
    return ts_gra


def bond_changes(
    gra: TransGraph,
) -> dict[BondKey, Change]:
    """Extract the formed and broken bonds from a transition graph."""
    change = nx.get_edge_attributes(gra, TransBond.change)
    return {k: v for k, v in change.items() if v is not None}


def formed_bonds(gra: TransGraph) -> set[BondKey]:
    """Extract the formed bonds from a transition graph."""
    changes = bond_changes(gra)
    return {k for k, v in changes.items() if v == Change.FORMED}


def broken_bonds(gra: TransGraph) -> set[BondKey]:
    """Extract the broken bonds from a transition graph."""
    changes = bond_changes(gra)
    return {k for k, v in changes.items() if v == Change.BROKEN}


def reverse(gra: TransGraph) -> TransGraph:
    """Reverse the direction of a transition graph."""
    changes = bond_changes(gra)
    changes = {
        k: Change.FORMED if v == Change.BROKEN else Change.BROKEN
        for k, v in changes.items()
    }
    return from_bond_changes(products_graph(gra), changes)


def reactants_graph(gra: TransGraph) -> MolGraph:
    """Extract the reactant graph from a transition graph."""
    rct_gra = MolGraph()
    rct_gra.add_nodes_from(gra.nodes(data=True))
    rct_gra.add_edges_from(gra.edges(data=True))
    rct_gra.remove_edges_from(formed_bonds(gra))
    return rct_gra


def products_graph(gra: TransGraph) -> MolGraph:
    """Extract the product graph from a transition graph."""
    prd_gra = MolGraph()
    prd_gra.add_nodes_from(gra.nodes(data=True))
    prd_gra.add_edges_from(gra.edges(data=True))
    prd_gra.remove_edges_from(broken_bonds(gra))
    return prd_gra
