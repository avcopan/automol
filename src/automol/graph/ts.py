"""Core molecular graph functions.

Uses NetworkX for graph representation, with Atom and Bond data validation.

Does not include bond order information.
"""

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import Any

import more_itertools as mit
import networkx as nx
from rdkit.Chem import rdchem

from .core import Atom, Bond, Graph, isomorphisms, remove_bonds


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


# From
def all_from_reactants_and_products(
    rct_gra: Graph[Atom, Bond],
    prd_gra: Graph[Atom, Bond],
    *,
    extra: int = 2,
    isomorphs: bool = False,
) -> Iterator[Graph[Atom, TransBond]]:
    """Fewest-bonds-first constructive count vector mappings.

    Parameters
    ----------
    rct_gra
        Reactant graph
    prd_gra
        Product graph
    extra
        Maximum number of additional breakable bonds to traverse, by default 2
    isomorphs
        Whether to retain isomorphs (True) or filter them out (False)

    Yields
    ------
        Transition state graphs
    """
    ccv = CCV(rct_gra, prd_gra)
    yield from (gra for gra, _ in ccv.filtered(extra=extra, isomorphs=isomorphs))


# Algorithms
def is_isomorphic[AtomT: Atom, BondT: Bond](
    gra1: Graph[AtomT, BondT], gra2: Graph[AtomT, BondT]
) -> bool:
    """Check if two graphs are isomorphic."""
    atom_fields = gra1.atom_type.model_fields.keys()
    bond_fields = gra1.bond_type.model_fields.keys()

    def atom_match(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
        return all(n1[field] == n2[field] for field in atom_fields)

    def bond_match(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        return all(e1[field] == e2[field] for field in bond_fields)

    return nx.is_isomorphic(gra1, gra2, node_match=atom_match, edge_match=bond_match)


# Transition state graphs
BondKey = tuple[int, int]
BondSymbol = tuple[str, str]
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


# Reaction mapping
@dataclass
class CCV:
    """CCV Reaction Mapping helper class."""

    reactants: Graph[Atom, Bond]
    products: Graph[Atom, Bond]
    _seen_bond_changes: list[dict[BondKey, Change]] = field(default_factory=list)

    @cached_property
    def reactant_bond_symbols(self) -> dict[BondKey, BondSymbol]:
        """Extract the bond symbols from the reactant graph."""
        return bond_symbols(self.reactants)

    @cached_property
    def product_bond_symbols(self) -> dict[BondKey, BondSymbol]:
        """Extract the bond symbols from the product graph."""
        return bond_symbols(self.products)

    @cached_property
    def reactant_bond_count_vector(self) -> Counter[BondSymbol]:
        """Extract the CCV bond count vector from the reactant graph."""
        return Counter(self.reactant_bond_symbols.values())

    @cached_property
    def product_bond_count_vector(self) -> Counter[BondSymbol]:
        """Extract the CCV bond count vector from the product graph."""
        return Counter(self.product_bond_symbols.values())

    def filtered(
        self, *, extra: int = 2, isomorphs: bool = False
    ) -> Iterator[tuple[Graph[Atom, TransBond], dict[int, int]]]:
        """Yield filtered CCV algorithm results.

        Parameters
        ----------
        extra
            Maximum number of additional breakable bonds to traverse, by default 2
        isomorphs
            Whether to retain isomorphs (True) or filter them out (False)
        bond_score
            Whether to assign bond scores and return only the lowest-scoring result(s)

        Yields
        ------
            Transition state graphs and reaction mappings
        """
        seen_gras = []
        for gra, mapping in self.results(extra=extra):
            if isomorphs or not any(is_isomorphic(gra, g) for g in seen_gras):
                seen_gras.append(gra)
                yield gra, mapping

    def results(
        self, *, extra: int = 2
    ) -> Iterator[tuple[Graph[Atom, TransBond], dict[int, int]]]:
        """Yield all CCV algorithm results.

        Parameters
        ----------
        extra
            Maximum number of additional breakable bonds to traverse, by default 2

        Yields
        ------
            Transition state graphs and reaction mappings
        """
        for num_extra in range(extra + 1):
            for extra_cv in self._extra_breaking_bond_count_vectors(num_extra):
                results = itertools.chain.from_iterable(
                    self._distinct_transition_graphs_with_reaction_mappings(b1, b2)
                    for b1, b2 in self._breaking_bond_patterns(extra_cv)
                )

                # If nothing was found, continue
                first_result = next(results, None)
                if first_result is None:
                    continue

                # Otherwise, return all results for this `extra_cv` and quit
                yield first_result
                yield from results
                return

    def _extra_breaking_bond_count_vectors(
        self, num: int
    ) -> Iterator[Counter[BondSymbol]]:
        """Iterate over bond count vectors for a given number of extra bonds."""
        rcv = self.reactant_bond_count_vector
        pcv = self.product_bond_count_vector
        cv = rcv - (rcv - pcv)
        symbs = sorted(cv.keys(), key=lambda x: (-x.count("H"), x))
        symbs_pool = list(
            itertools.chain.from_iterable(itertools.repeat(s, cv[s]) for s in symbs)
        )
        return map(
            Counter, mit.unique_everseen(itertools.combinations(symbs_pool, num))
        )

    def _breaking_bond_patterns(
        self, extra_cv: Counter[BondSymbol]
    ) -> Iterator[tuple[tuple[BondKey, ...], tuple[BondKey, ...]]]:
        """Iterate over bond combinations consistent with a given bond count vector."""
        rcv0 = self.reactant_bond_count_vector
        pcv0 = self.product_bond_count_vector
        rcv = (rcv0 - pcv0) + extra_cv
        pcv = (pcv0 - rcv0) + extra_cv
        return itertools.product(
            count_vector_bond_combinations(
                self.reactants, rcv, self.reactant_bond_symbols
            ),
            count_vector_bond_combinations(
                self.products, pcv, self.product_bond_symbols
            ),
        )

    def _distinct_transition_graphs_with_reaction_mappings(
        self, break_bonds1: Sequence[BondKey], break_bonds2: Sequence[BondKey]
    ) -> Iterator[tuple[Graph[Atom, TransBond], dict[int, int]]]:
        """Iterate over reverse_isomorphisms with distinct bond changes."""
        gra1 = remove_bonds(self.reactants, break_bonds1)
        gra2 = remove_bonds(self.products, break_bonds2)
        for mapping in isomorphisms(gra1, gra2):
            bond_changes = bond_changes_from_mapping_and_break_pattern(
                mapping, break_bonds1, break_bonds2
            )
            if bond_changes not in self._seen_bond_changes:
                self._seen_bond_changes.append(bond_changes)
                gra = from_bond_changes(self.reactants, bond_changes)
                yield gra, mapping


def bond_symbols(G: Graph) -> dict[BondKey, BondSymbol]:  # noqa: N803
    """Get bond symbols by bond key."""
    return {
        (key1, key2): tuple(
            sorted([G.nodes[key1][Atom.symbol], G.nodes[key2][Atom.symbol]])
        )
        for key1, key2 in G.edges()
    }


def bond_symbol_keys(
    G: Graph,  # noqa: N803
    symbs: dict[BondKey, BondSymbol] | None = None,
) -> dict[BondSymbol, set[BondKey]]:
    """Get bond keys by bond symbol."""
    symbs = bond_symbols(G) if symbs is None else symbs
    symb_keys = defaultdict(set)
    for key, symb in symbs.items():
        symb_keys[symb].add(key)
    return dict(symb_keys)


def count_vector_bond_combinations(
    G: Graph,  # noqa: N803
    cv: Counter[BondSymbol],
    symbs: dict[BondKey, BondSymbol] | None = None,
) -> Iterator[tuple[BondKey, ...]]:
    """Iterate over bond combinations consistent with a given bond count vector."""
    symbs = bond_symbols(G) if symbs is None else symbs
    symb_keys = bond_symbol_keys(G, symbs)
    per_symbol_combination_iters = [
        itertools.combinations(symb_keys[symb], count) for symb, count in cv.items()
    ]
    for per_symbol_combinations in itertools.product(*per_symbol_combination_iters):
        yield tuple(itertools.chain.from_iterable(per_symbol_combinations))


def bond_changes_from_mapping_and_break_pattern(
    mapping: dict[int, int],
    break_bonds1: Sequence[BondKey],
    break_bonds2: Sequence[BondKey],
) -> dict[BondKey, Change]:
    """Construct a bond change dictionary from breaking and forming bond lists."""
    rev_map = {v: k for k, v in mapping.items()}
    break_bonds = break_bonds1
    form_bonds = {
        (rev_map[b[0]], rev_map[b[1]])
        if b[0] < b[1]
        else (rev_map[b[1]], rev_map[b[0]])
        for b in break_bonds2
    }
    return {
        **dict.fromkeys(break_bonds, Change.BROKEN),
        **dict.fromkeys(form_bonds, Change.FORMED),
    }
