"""automol tests."""

import numpy as np

import automol


def test__geometry() -> None:
    """Test the Geometry model."""
    geo = automol.Geometry(
        symbols=["O", "H", "H"], coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )
    assert geo.symbols == ["O", "H", "H"]
    assert np.array_equal(geo.coordinates, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
