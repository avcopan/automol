"""Vibrational normal-mode and frequency analysis tests."""

import numpy as np
import pytest

from automol import Geometry, geom
from automol.utils import constants


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )


def test__mass_weight_vector(water: Geometry) -> None:
    """Test the mass-weighting vector against per-atom masses."""
    vec = geom.mass_weight_vector(water)
    assert vec.shape == (9,)
    assert np.allclose(vec, np.sqrt(np.repeat(water.masses, 3)))


def test__translational_rotational_normal_modes_shapes(water: Geometry) -> None:
    """Test translational/rotational normal mode shapes and independence."""
    trans = geom.translational_normal_modes(water)
    rot = geom.rotational_normal_modes(water)
    assert trans.shape == (9, 3)
    assert rot.shape == (9, 3)

    combined = np.hstack([trans, rot])
    assert np.linalg.matrix_rank(combined) == 6  # noqa: PLR2004


def test__normal_mode_projection_default(water: Geometry) -> None:
    """Test that the default projection removes translation and rotation."""
    proj = geom.normal_mode_projection(water)
    assert proj.shape == (9, 3)


def test__normal_mode_projection_keep_all(water: Geometry) -> None:
    """Test that keeping both trans and rot modes returns the identity."""
    proj = geom.normal_mode_projection(water, trans=True, rot=True)
    assert np.allclose(proj, np.eye(9))


def test__vibrational_analysis_shapes(water: Geometry) -> None:
    """Test vibrational_analysis output shapes for a toy Hessian."""
    freqs, modes = geom.vibrational_analysis(water, np.eye(9))
    assert len(freqs) == 3  # noqa: PLR2004
    assert modes.shape == (9, 3)


def test__vibrational_analysis_wrong_hessian_shape_raises(water: Geometry) -> None:
    """Test that a malformed Hessian raises ValueError."""
    with pytest.raises(ValueError, match="is not"):
        geom.vibrational_analysis(water, np.zeros((4, 4)))


def test__harmonic_zpv_freqs_shortcut(water: Geometry) -> None:
    """Test that harmonic_zpv sums only positive frequencies when freqs is given."""
    value = geom.harmonic_zpv(water, hess=[], freqs=(100.0, 200.0, -50.0))
    expected = 0.5 * (100.0 + 200.0) * constants.WAVENUMBER_TO_HARTREE
    assert np.isclose(value, expected)


def test__harmonic_zpv_from_hessian(water: Geometry) -> None:
    """Test that harmonic_zpv without freqs falls back to vibrational_analysis."""
    hess = np.eye(9).tolist()
    freqs, _ = geom.vibrational_analysis(water, np.array(hess))
    expected = 0.5 * sum(f for f in freqs if f > 0.0) * constants.WAVENUMBER_TO_HARTREE
    value = geom.harmonic_zpv(water, hess)
    assert np.isclose(value, expected)
