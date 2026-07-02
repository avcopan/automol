"""Geometry canonicalization functions."""

import itertools
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from ..graph.core import Atom, Bond, MolGraph
from .transform import eckart_frame

if TYPE_CHECKING:
    from .core import Geometry


def canonical_frame(geo: "Geometry", *, decimals: int = 4) -> "Geometry":
    """Return a copy of the geometry in a canonical, invariant Eckart frame.

    The geometry is first moved to its center of mass and aligned to its principal axes
    of inertia (Eckart frame). Sign ambiguity is distinguished by evaluating the four
    sign combinations that preserve a right-handed frame and selecting the one producing
    the smallest rounded coordinate sum.

    Parameters
    ----------
    geo
        Geometry.
    decimals
        Number of decimal places to round coordinates before comparing candidate frames.

    Returns
    -------
    Geometry
        Geometry in a canonical frame.
    """
    eckart_geo = eckart_frame(geo, in_place=False)

    best = None
    best_key: tuple[int, tuple[int, ...]] | None = None

    # Evaluate the sign combinations that preserve a right-handed frame
    for sx, sy in itertools.product((1.0, -1.0), repeat=2):
        signs = np.array([sx, sy, sx * sy])
        candidate_coords = eckart_geo.coordinates * signs
        icoords = np.rint(candidate_coords * 10**decimals).astype(np.int64)

        # Tie-break by minimum coordinate sum then flattened array order
        key = (int(np.sum(icoords)), tuple(icoords.flatten()))
        if best_key is None or key < best_key:
            best_key = key
            best = eckart_geo.model_copy(deep=True)
            best.coordinates = candidate_coords

    if best:
        return best

    msg = f"Best canonical frame candidate not determined for {geo = }."
    raise ValueError(msg)


def canonical_sorting(
    geo: "Geometry",
    *,
    delta: float = 1.1,
    decimals: int = 4,
    truncate: bool = False,
) -> list[int]:
    """Sort atoms using spatial orientations, graph connectivity, and Hill priorities.

    Canonical sorting order:
    1. Canonicalize spatial orientations using `canonical_frame(...)` to resolve frame
    sign ambiguity.
    2. Construct a molecular graph from the invariant coordinates, filtering out
    hypervalent connectivity (*optional*).
    3. Determine the optimal parent chain based on absolute path length with terminal
    nodes given highest priority for the 0-th index and tie-breaks broken by minimizing
    locants then atomic number.
    4. Traverse sequentially outward from the parent chain across all juction branches
    to sort subcomponents by max branch length, atomic number, and spatial coordinates.
    5. Order hydrogens according to the rank of their corresponding heavy atoms then
    locally by spatial coordinates.
    6. Map the final structural path sequence to traditional Hill sorting priorities
    (Carbon, then Hydrogen, followed by remaining atoms sorted alphabetically), using
    the structural canononical index rank to resolve ties.

    Parameters
    ----------
    geo
        Geometry.
    delta
        Factor to scale covalent matrices for bond consideration.
        `bond_cutoff = delta * (r_covalent1 + r_covalent2)`
    decimals
        Number of decimal places to consider in tie-breaking.
    truncate
        If True, truncate hypervalent bonds by highest covalent radius deviation.

    Returns
    -------
    list[int]
        Indices for canonical sorting.
    """
    canonical_geo = canonical_frame(geo, decimals=decimals)
    gra = _molecular_graph(canonical_geo, delta=delta, truncate=truncate)
    parent = _best_parent(gra, decimals=decimals)

    heavy_idxs = []
    visited = set()

    def traverse(node: int) -> None:
        if node in visited:
            return

        visited.add(node)
        heavy_idxs.append(node)
        for branch_start in _sort_junction_branches(gra, node, parent):
            traverse(branch_start)

    for node in parent:
        traverse(node)

    ordered_idxs = list(heavy_idxs)

    # Sort and append hydrogens locally by neighbor coordinate order
    for heavy in heavy_idxs:
        hydrogens = [
            n
            for n in gra.neighbors(heavy)
            if not Atom.model_validate(gra.nodes[n]).is_heavy
        ]

        def sorting_key(idx: int) -> tuple[int, int, int, int]:
            c: np.ndarray = np.rint(gra.nodes[idx]["coords"] * 10**decimals)
            return (c[0], c[1], c[2], idx)

        ordered_idxs.extend(sorted(hydrogens, key=sorting_key))

    # Find atoms not reached by bond determination
    missing = sorted(
        set(gra.nodes) - set(ordered_idxs),
        key=lambda idx: tuple(np.rint(gra.nodes[idx][Atom.coords] * 10**decimals)),
    )
    if len(missing) > 0:
        msg = (
            f"{len(missing)} atom(s) not picked up by bond determination.\n"
            f"{missing = }.\nConsider increasing delta ({delta = })."
        )
        raise ValueError(msg)

    rank = {node_idx: rank for rank, node_idx in enumerate(ordered_idxs)}

    def hill_sorting_key(idx: int) -> tuple[int, str, int]:
        symbol = gra.nodes[idx][Atom.symbol]
        priority = 0 if symbol == "C" else (1 if symbol == "H" else 2)
        return (priority, symbol, rank[idx])

    return sorted(ordered_idxs, key=hill_sorting_key)


def _molecular_graph(
    geo: "Geometry", *, delta: float = 1.05, truncate: bool = False
) -> MolGraph:
    """Construct a molecular graph from geometry coordinates based on covalent radii.

    Parameters
    ----------
    geo
        Geometry
    delta
        Factor to scale covalent matrices for bond consideration.
        `bond_cutoff = delta * (r_covalent1 + r_covalent2)`
    truncate
        If True, truncate hypervalent bonds by highest covalent radius deviation.

    Returns
    -------
    nx.Graph
        Molecular graph.
    """
    gra = MolGraph(atom_type=Atom, bond_type=Bond)

    for i in range(geo.atom_count):
        atom = Atom(symbol=geo.symbols[i], coords=np.array(geo.coordinates[i]))
        gra.add_node(i, **atom.model_dump())

    for n1, n2 in itertools.combinations(gra.nodes, 2):
        at1 = Atom.model_validate(gra.nodes[n1])
        at2 = Atom.model_validate(gra.nodes[n2])

        if at1.coords is None or at2.coords is None:
            msg = "Cannot determine bonds without atomic coordinates."
            raise ValueError(msg)

        dist = np.linalg.norm(at1.coords - at2.coords)
        if dist <= delta * (at1.covalent_radius + at2.covalent_radius):
            gra.add_edge(n1, n2, **Bond(distance=float(dist)).model_dump())

    return _truncate_bonds(gra) if truncate else gra


def _heavy_subgraph(gra: MolGraph) -> MolGraph:
    """Return a copy of the subgraph containing only heavy atoms.

    Parameters
    ----------
    gra
        Molecular graph.

    Returns
    -------
    MolGraph
        Heavy atom graph.
    """
    heavy_nodes = [n for n in gra.nodes if Atom.model_validate(gra.nodes[n]).is_heavy]
    return gra.subgraph(heavy_nodes).copy()


def _path_scoring_key(
    path: list[int], heavy_gra: MolGraph, terminals: list[int], decimals: int
) -> tuple:
    """Compute the tie-breaking key for a specific ordered path direction."""
    terminal_priority = 0 if path[0] in terminals else 1

    # Find and sort locants (substituent/branch positions)
    locants = []
    path_set = set(path)
    for idx, node in enumerate(path):
        neighbors = set(heavy_gra.neighbors(node))
        if neighbors - path_set:
            locants.append(idx)
    locants.sort()

    # Priority sequences
    carbons_seq = [0 if heavy_gra.nodes[n][Atom.symbol] == "C" else 1 for n in path]
    z_seq = [Atom.model_validate(heavy_gra.nodes[n]).atomic_number for n in path]

    # Coordinate mapping
    coords_seq = []
    for n in path:
        icoords = np.rint(heavy_gra.nodes[n]["coords"] * 10**decimals)
        coords_seq.extend(icoords.tolist())

    return (terminal_priority, locants, carbons_seq, z_seq, coords_seq)


def _best_parent(gra: MolGraph, *, decimals: int = 4) -> list[int]:
    """Find the longest backbone paths and tie-break using locant minimization."""
    heavy_gra = _heavy_subgraph(gra)

    if len(heavy_gra) == 1:
        return list(heavy_gra.nodes)

    terminals = [n for n in heavy_gra.nodes if heavy_gra.degree(n) == 1]

    # 1. Generate all possible paths based on terminal counts
    if len(terminals) > 1:
        paths = []
        for n1, n2 in itertools.combinations(terminals, 2):
            try:
                paths.append(nx.shortest_path(heavy_gra, n1, n2))
            except nx.NetworkXNoPath:
                continue
    elif len(terminals) == 1:
        paths = list(nx.single_source_shortest_path(heavy_gra, terminals[0]).values())
    else:
        paths = [
            p
            for _, targets in nx.all_pairs_shortest_path(heavy_gra)
            for p in targets.values()
        ]

    if not paths:
        msg = "Could not determine best backbone path."
        raise ValueError(msg)

    # 2. Filter down to only the longest paths
    max_len = max(len(p) for p in paths)
    candidate_paths = [p for p in paths if len(p) == max_len]

    # 3. Consider both directions for each candidate path
    all_directional_paths = [
        dir_path for path in candidate_paths for dir_path in (path, path[::-1])
    ]

    # 4. Tie-break using Python's min with our scoring helper
    best_path = min(
        all_directional_paths,
        key=lambda p: _path_scoring_key(p, heavy_gra, terminals, decimals),
        default=None,
    )

    if best_path is None:
        msg = "Could not determine best backbone path."
        raise ValueError(msg)

    return best_path


def _max_branch_length(gra: MolGraph, start_node: int, parent_node: int) -> int:
    """Determine max path length of a branch.

    Parameters
    ----------
    gra_heavy
        Molecular graph.
    start_node
        Index of the first branch node.
    parent_node
        Index of the node on the parent chain.

    Returns
    -------
    int
        Maximum length of the branch.
    """
    gra = gra.copy()
    gra.remove_edge(start_node, parent_node)

    try:
        subgra = gra.subcomponent(start_node)

    except AttributeError:
        subgra = nx.node_connected_component(gra, start_node)

    lens = nx.single_source_shortest_path_length(gra.subgraph(subgra), start_node)
    return max(lens.values()) if lens else 0


def _sort_junction_branches(
    gra: MolGraph, junction_node: int, parent_chain: list[int], *, decimals: int = 4
) -> list[int]:
    """Sort side branches at a junction by length, then atomic symbol, then coordinates.

    Parameters
    ----------
    gra
        Molecular graph.
    junction
        Index of the parent chain junction.
    backbone
        Indices of the parent chain.
    decimals
        Decimal places to consider in tie-breaking.

    Returns
    -------
    list[int]
        Sorted list of indices for branch starts.
    """
    heavy_gra = _heavy_subgraph(gra)
    neighbors = set(heavy_gra.neighbors(junction_node))
    branch_starts = list(neighbors - set(parent_chain))

    def sorting_key(node: int) -> tuple[int, str, int, int, int, int]:
        length = _max_branch_length(gra, node, junction_node)
        symbol = heavy_gra.nodes[node][Atom.symbol]
        icoords = np.rint(heavy_gra.nodes[node]["coords"] * 10**decimals)
        return (-length, symbol, icoords[0], icoords[1], icoords[2], node)

    return sorted(branch_starts, key=sorting_key)


def _truncate_bonds(gra: MolGraph) -> MolGraph:
    """Truncate hypervalent bonds by highest covalent radius deviation.

    Parameters
    ----------
    gra
        Molecular graph.

    Returns
    -------
    MolGraph
        Molecular graph with hypervalencies truncated.
    """
    gra = gra.copy()

    while True:
        hypervalencies = []
        for n in gra.nodes:
            # Step 1: Safely validate the dictionary data into an Atom instance
            at = Atom.model_validate(gra.nodes[n])
            # Step 2: Use the property directly
            if gra.degree(n) > at.valence:
                hypervalencies.append(n)

        if not hypervalencies:
            break

        worst = None
        max_dev = float("-inf")

        for u in hypervalencies:
            at1 = Atom.model_validate(gra.nodes[u])

            for v in gra.neighbors(u):
                at2 = Atom.model_validate(gra.nodes[v])

                edge_data = gra.get_edge_data(u, v)
                bond = Bond.model_validate(edge_data)

                if not bond.distance:
                    msg = "Cannot manage hypervalencies without bond distances."
                    raise ValueError(msg)

                # Deviation is how far OVER the covalent limit it is stretched
                deviation = bond.distance - (at1.covalent_radius + at2.covalent_radius)

                if deviation > max_dev:
                    max_dev = deviation
                    worst = (u, v)

        if worst:
            gra.remove_edge(*worst)
        else:
            break

    return gra
