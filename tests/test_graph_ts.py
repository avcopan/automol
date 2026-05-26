"""Graph tests."""

import pytest

from automol import graph
from automol.graph import ts
from automol.graph.ts import CCV


@pytest.mark.parametrize(
    ("rct_smi", "prd_smi", "ts_count"),
    [
        ("CCO", "[CH2]O.[CH3]", 1),
        ("CC.[OH]", "C[CH2].O", 1),
        ("CC.[CH3]", "C[CH2].C", 1),
        ("CCO[O]", "[CH2]COO", 1),
        ("CCO[O]", "C=C.O[O]", 1),
    ],
)
def test__all_from_reactants_and_products(
    rct_smi: str, prd_smi: str, ts_count: int
) -> None:
    """Test transition state graph generation from reactants and products."""
    rct_gra0 = graph.from_smiles(rct_smi)
    prd_gra0 = graph.from_smiles(prd_smi)
    ts_gras = list(ts.all_from_reactants_and_products(rct_gra0, prd_gra0))
    assert len(ts_gras) == ts_count
    for ts_gra in ts_gras:
        rct_gra = ts.reactants_graph(ts_gra)
        prd_gra = ts.products_graph(ts_gra)
        assert graph.is_isomorphic(rct_gra, rct_gra0)
        assert graph.is_isomorphic(prd_gra, prd_gra0)


@pytest.mark.parametrize(
    (
        "rct_smi",
        "prd_smi",
        "unique_isomorphs",
        "unique_bond_changes",
        "maximum_bond_score",
        "result_count",
    ),
    [
        ("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]", True, True, True, 2),
        ("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]", True, True, False, 4),
        ("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]", False, True, True, 4),
        ("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]", False, False, True, 16),
    ],
)
def test__ccv_reaction_mapping(  # noqa: PLR0913
    rct_smi: str,
    prd_smi: str,
    *,
    unique_isomorphs: bool,
    unique_bond_changes: bool,
    maximum_bond_score: bool,
    result_count: int,
) -> None:
    """Test transition state graph generation from reactants and products."""
    rct_gra0 = graph.from_smiles(rct_smi)
    prd_gra0 = graph.from_smiles(prd_smi)
    ccv = CCV(reactants=rct_gra0, products=prd_gra0)
    results = list(
        ccv.results(
            unique_isomorphs=unique_isomorphs,
            unique_bond_changes=unique_bond_changes,
            maximum_bond_score=maximum_bond_score,
        )
    )
    assert len(results) == result_count
