"""Graph tests."""

from automol import graph


def test__inchi() -> None:
    """Test graph inchi."""
    water_inchi = "InChI=1S/H2O/h1H2"
    water = graph.from_inchi(water_inchi)
    assert graph.inchi(water) == water_inchi


def test__transition_graphs() -> None:
    """Test transition graph construction."""
    RG0 = graph.from_smiles("CCO")  # noqa: N806
    PG0 = graph.from_smiles("[CH3].[CH2]O")  # noqa: N806
    TGs = graph.transition_state_graphs(RG0, PG0)  # noqa: N806
    for TG in TGs:  # noqa: N806
        RG = graph.reactants_graph(TG)  # noqa: N806
        PG = graph.products_graph(TG)  # noqa: N806
        assert graph.is_isomorphic(RG, RG0)
        assert graph.is_isomorphic(PG, PG0)
