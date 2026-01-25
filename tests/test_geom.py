"""geom tests."""

import numpy as np
import pytest

from automol import Geometry, geom


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"], coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )


def test__center_of_mass(water: Geometry) -> None:
    """Test center of mass."""
    assert np.allclose(geom.center_of_mass(water), [0.05595744, 0.05595744, 0.0])
