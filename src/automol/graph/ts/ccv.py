"""Constructive Count Vector (CCV) reaction mapping algorithm."""

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property

import more_itertools as mit

from ..core import (
    Atom,
    BondKey,
    Graph,
    MolGraph,
    is_isomorphic,
    isomorphisms,
    open_valences,
    remove_bonds,
)
from .core import (
    Change,
    TransGraph,
    formed_bonds,
    from_bond_changes,
    reactants_graph,
    reverse,
)

BondSymbol = tuple[str, str]


# From
def all_from_reactants_and_products(
    rct_gra: MolGraph, prd_gra: MolGraph
) -> Iterator[TransGraph]:
    """Fewest-bonds-first constructive count vector mappings.

    Parameters
    ----------
    rct_gra
        Reactant graph
    prd_gra
        Product graph

    Yields
    ------
        Transition state graphs
    """
    ccv = CCV(rct_gra, prd_gra)
    yield from (gra for gra, _ in ccv.results())


# Reaction mapping
@dataclass
class CCV:
    """CCV Reaction Mapping helper class."""

    reactants: MolGraph
    products: MolGraph

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

    def results(
        self,
        *,
        extra_breaks: int = 2,
        unique_isomorphs: bool = True,
        unique_bond_changes: bool = True,
        maximum_bond_score: bool = True,
    ) -> Iterator[tuple[TransGraph, dict[int, int]]]:
        """Yield CCV algorithm results, with optional filtering.

        Parameters
        ----------
        extra_breaks
            Maximum number of additional breakable bonds to traverse, by default 2
        unique_isomorphs
            Whether to include only the unique isomorphs
        unique_bond_changes
            Whether to include only results with unique bond changes
        maximum_bond_score
            Whether to assign bond scores and include only results with the
            maximum score.  Higher scores use more open valences (radical sites
            and double bonds) for bond formation.

        Yields
        ------
            Transition state graphs and reaction mappings
        """
        # Determine maximum bond score if filtering by bond score
        max_score = None
        if maximum_bond_score:
            max_score = max(
                bonding_score(gra)
                for gra, _ in self.all_results(
                    extra_breaks=extra_breaks, unique_bond_changes=True
                )
            )

        seen_gras = []
        for gra, mapping in self.all_results(
            extra_breaks=extra_breaks, unique_bond_changes=unique_bond_changes
        ):
            score = bonding_score(gra) if maximum_bond_score else None

            # 1. Check for maximum bond score if filtering by bond score
            if maximum_bond_score and (score != max_score):
                continue

            # 2. Check for isomorphism if removing isomorphs
            if unique_isomorphs and any(is_isomorphic(gra, g) for g in seen_gras):
                continue

            seen_gras.append(gra)
            yield gra, mapping

    def all_results(
        self, *, extra_breaks: int = 2, unique_bond_changes: bool = False
    ) -> Iterator[tuple[TransGraph, dict[int, int]]]:
        """Yield all CCV algorithm results.

        Parameters
        ----------
        extra_breaks
            Maximum number of additional breakable bonds to traverse, by default 2
        unique_bond_changes
            Whether to include only results with unique bond changes

        Yields
        ------
            Transition state graphs and reaction mappings
        """
        seen_changes = [] if unique_bond_changes else None
        for num_extra in range(extra_breaks + 1):
            for extra_cv in self._extra_breaking_bond_count_vectors(num_extra):
                results = itertools.chain.from_iterable(
                    self._distinct_transition_graphs_with_reaction_mappings(
                        b1, b2, seen_changes=seen_changes
                    )
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
        self,
        break_bonds1: Sequence[BondKey],
        break_bonds2: Sequence[BondKey],
        seen_changes: list[dict[BondKey, Change]] | None = None,
    ) -> Iterator[tuple[TransGraph, dict[int, int]]]:
        """Iterate over reverse_isomorphisms with distinct bond changes."""
        gra1 = remove_bonds(self.reactants, break_bonds1)
        gra2 = remove_bonds(self.products, break_bonds2)
        for mapping in isomorphisms(gra1, gra2):
            changes = bond_changes_from_mapping_and_break_pattern(
                mapping, break_bonds1, break_bonds2
            )
            if seen_changes is not None:
                if changes in seen_changes:
                    continue

                seen_changes.append(changes)

            gra = from_bond_changes(self.reactants, changes)
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


def forward_bonding_score(gra: TransGraph) -> int:
    """Calculate the forward bonding score for a graph.

    Defined as the sum of the open valences of the atoms involved in forming
    bonds, from the reactants side.
    """
    rgra = reactants_graph(gra)
    form_keys = list(set(itertools.chain.from_iterable(formed_bonds(gra))))
    return sum(open_valences(rgra, form_keys))


def bonding_score(gra: TransGraph) -> int:
    """Calculate the total bonding score for a graph.

    Defined as the sum of forward and reverse bonding scores.
    """
    return forward_bonding_score(gra) + forward_bonding_score(reverse(gra))
