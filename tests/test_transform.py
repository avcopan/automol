"""Geometry transform tests."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from automol import Geometry, geom


def test__reflection(peroxide: Geometry) -> None:
    """Test reflection."""
    normal = np.random.rand(3)  # noqa: NPY002
    refl_peroxide = geom.transform.reflect(peroxide, normal)
    double_refl_peroxide = geom.transform.reflect(refl_peroxide, normal)
    assert not np.allclose(peroxide.coordinates, refl_peroxide.coordinates)
    assert np.allclose(
        peroxide.coordinates, double_refl_peroxide.coordinates, atol=1e-7
    )


def test__translate(water: Geometry) -> None:
    """Test translation."""
    shift = [1.0, 2.0, 3.0]
    translated = geom.transform.translate(water, shift)
    assert np.allclose(translated.coordinates, water.coordinates + shift)
    assert not np.allclose(translated.coordinates, water.coordinates)


def test__translate_in_place(water: Geometry) -> None:
    """Test in-place translation."""
    original = water.coordinates.copy()
    result = geom.transform.translate(water, [1.0, 0.0, 0.0], in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, np.add(original, [1.0, 0.0, 0.0]))


def test__translate_with_keys(water: Geometry) -> None:
    """Test translation of a subset of atoms."""
    original = water.coordinates.copy()
    translated = geom.transform.translate(water, [1.0, 0.0, 0.0], keys=[0])
    assert np.allclose(translated.coordinates[0], original[0] + [1.0, 0.0, 0.0])
    assert np.allclose(translated.coordinates[1:], original[1:])


def test__rotate(water: Geometry) -> None:
    """Test rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    rotated = geom.transform.rotate(water, rot)
    assert np.allclose(rotated.coordinates, rot.apply(water.coordinates))
    dist_before = geom.distance_matrix(water)
    dist_after = geom.distance_matrix(rotated)
    assert np.allclose(dist_before, dist_after)


def test__rotate_in_place(water: Geometry) -> None:
    """Test in-place rotation."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    expected = rot.apply(water.coordinates)
    result = geom.transform.rotate(water, rot, in_place=True)
    assert result is water
    assert np.allclose(water.coordinates, expected)


def test__transition_raises_for_mismatched_spin(water: Geometry) -> None:
    """Test that transition() rejects geometries with different spins."""
    water_triplet = water.model_copy(update={"spin": 2})
    assert water.spin != water_triplet.spin
    with pytest.raises(ValueError, match="spin"):
        geom.transform.transition(water, water_triplet)


def test__transition_identity(water: Geometry) -> None:
    """Test the (degenerate) transition between a geometry and itself."""
    ts_geo = geom.transform.transition(water, water)
    assert ts_geo.symbols == water.symbols
    assert ts_geo.spin == water.spin
    assert ts_geo.coordinates.shape == water.coordinates.shape
    assert np.all(np.isfinite(ts_geo.coordinates))


def test__transition_hydrogen_abstraction() -> None:
    """Test the transition geometry for an H-abstraction reaction.

    F-H + Cl -> F + H-Cl: the migrating H atom breaks its bond to F and forms
    a new bond to Cl.
    """
    reactant = Geometry(
        symbols=["F", "H", "Cl"],
        coordinates=[[0, 0, 0], [0.92, 0, 0], [3.5, 0, 0]],
        charge=0,
        spin=0,
    )
    product = Geometry(
        symbols=["F", "H", "Cl"],
        coordinates=[[0, 0, 0], [3.0, 0, 0], [4.27, 0, 0]],
        charge=0,
        spin=0,
    )
    ts_geo = geom.transform.transition(reactant, product)
    assert ts_geo.symbols == reactant.symbols
    assert ts_geo.spin == reactant.spin
    assert ts_geo.coordinates.shape == reactant.coordinates.shape
    assert np.all(np.isfinite(ts_geo.coordinates))
