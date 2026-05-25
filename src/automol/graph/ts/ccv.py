"""Constructive Count Vector (CCV) reaction mapping algorithm."""

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property

import more_itertools as mit

from ..core import (
    Atom,
    BondKey,
    Graph,
    MolGraph,
    is_isomorphic,
    isomorphisms,
    remove_bonds,
)
from .core import Change, TransGraph, from_bond_changes

BondSymbol = tuple[str, str]


# From
def all_from_reactants_and_products(
    rct_gra: MolGraph,
    prd_gra: MolGraph,
    *,
    extra: int = 2,
    isomorphs: bool = False,
) -> Iterator[TransGraph]:
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


# Reaction mapping
@dataclass
class CCV:
    """CCV Reaction Mapping helper class."""

    reactants: MolGraph
    products: MolGraph
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
    ) -> Iterator[tuple[TransGraph, dict[int, int]]]:
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

    def results(self, *, extra: int = 2) -> Iterator[tuple[TransGraph, dict[int, int]]]:
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
    ) -> Iterator[tuple[TransGraph, dict[int, int]]]:
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
