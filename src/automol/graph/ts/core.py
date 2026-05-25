"""Core molecular graph functions.

Uses NetworkX for graph representation, with Atom and Bond data validation.

Does not include bond order information.
"""

from enum import StrEnum

import networkx as nx
from rdkit.Chem import rdchem

from ..core import Atom, Bond, BondKey, Graph


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
FORMED_BOND = TransBond(change=Change.FORMED)
BROKEN_BOND = TransBond(change=Change.BROKEN)


def from_bond_changes(
    gra: Graph[Atom, Bond], bond_changes: dict[BondKey, Change]
) -> Graph[Atom, TransBond]:
    """Construct a transition graph from a graph and bond changes."""
    ts_gra = Graph(atom_type=Atom, bond_type=TransBond)
    ts_gra.add_nodes_from(gra.nodes(data=True))
    ts_gra.add_edges_from(gra.edges(), change=None)
    formed_bonds = {k for k, c in bond_changes.items() if c == Change.FORMED}
    broken_bonds = {k for k, c in bond_changes.items() if c == Change.BROKEN}
    ts_gra.add_edges_from(formed_bonds, change=Change.FORMED)
    ts_gra.add_edges_from(broken_bonds, change=Change.BROKEN)
    ts_gra.validate()
    return ts_gra


def bond_changes(
    gra: Graph[Atom, TransBond],
) -> dict[BondKey, Change]:
    """Extract the formed and broken bonds from a transition graph."""
    change = nx.get_edge_attributes(gra, TransBond.change)
    return {k: v for k, v in change.items() if v is not None}


def formed_bonds(gra: Graph[Atom, TransBond]) -> set[BondKey]:
    """Extract the formed bonds from a transition graph."""
    changes = bond_changes(gra)
    return {k for k, v in changes.items() if v == Change.FORMED}


def broken_bonds(gra: Graph[Atom, TransBond]) -> set[BondKey]:
    """Extract the broken bonds from a transition graph."""
    changes = bond_changes(gra)
    return {k for k, v in changes.items() if v == Change.BROKEN}


def reverse(gra: Graph[Atom, TransBond]) -> Graph[Atom, TransBond]:
    """Reverse the direction of a transition graph."""
    changes = bond_changes(gra)
    changes = {
        k: Change.FORMED if v == Change.BROKEN else Change.BROKEN
        for k, v in changes.items()
    }
    return from_bond_changes(gra, changes)


def reactants_graph(gra: Graph[Atom, TransBond]) -> Graph[Atom, Bond]:
    """Extract the reactant graph from a transition graph."""
    rct_gra = Graph(atom_type=Atom, bond_type=Bond)
    rct_gra.add_nodes_from(gra.nodes(data=True))
    rct_gra.add_edges_from(gra.edges(data=True))
    rct_gra.remove_edges_from(formed_bonds(gra))
    return rct_gra


def products_graph(gra: Graph[Atom, TransBond]) -> Graph[Atom, Bond]:
    """Extract the product graph from a transition graph."""
    prd_gra = Graph(atom_type=Atom, bond_type=Bond)
    prd_gra.add_nodes_from(gra.nodes(data=True))
    prd_gra.add_edges_from(gra.edges(data=True))
    prd_gra.remove_edges_from(broken_bonds(gra))
    return prd_gra
