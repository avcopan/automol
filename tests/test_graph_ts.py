"""Graph tests."""

import pytest

from automol import graph
from automol.graph import ts


@pytest.mark.parametrize(
    ("rct_smi", "prd_smi", "ts_count"),
    [
        ("CCO", "[CH2]O.[CH3]", 1),
        ("CC.[OH]", "C[CH2].O", 1),
        ("CC.[CH3]", "C[CH2].C", 1),
        ("CCO[O]", "[CH2]COO", 1),
        ("CCO[O]", "C=C.O[O]", 2),
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
