"""Geometry transform tests."""

import numpy as np
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
