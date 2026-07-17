"""Geometry comparison tests."""

from scipy.spatial.transform import Rotation

from automol import Geometry, geom


def test__is_duplicate_conformer_rotated_translated_match(water: Geometry) -> None:
    """Test that a rotated and translated copy is recognized as a duplicate."""
    rot = Rotation.from_euler("z", 60, degrees=True)
    duplicate = geom.transform.rotate(water, rot)
    duplicate = geom.transform.translate(duplicate, [5.0, 5.0, 5.0])

    assert all(geom.is_duplicate_conformer(water, [duplicate]))


def test__is_duplicate_conformer_different_conformer_no_match(water: Geometry) -> None:
    """Test that a geometrically distinct conformer is not a duplicate."""
    stretched = water.model_copy(deep=True)
    stretched.coordinates[1] *= 1.5

    assert not any(geom.is_duplicate_conformer(water, [stretched]))


def test__is_duplicate_conformer_different_atom_count_no_match(
    water: Geometry, peroxide: Geometry
) -> None:
    """Test that a different atom count never counts as a duplicate."""
    assert not any(geom.is_duplicate_conformer(water, [peroxide]))


def test__is_duplicate_conformer_empty_list_no_match(water: Geometry) -> None:
    """Test that an empty candidate list never matches."""
    assert not any(geom.is_duplicate_conformer(water, []))


def test__is_duplicate_conformer_custom_threshold(water: Geometry) -> None:
    """Test that a tighter threshold rejects an otherwise-matching candidate."""
    stretched = water.model_copy(deep=True)
    stretched.coordinates[1] *= 1.01

    assert all(geom.is_duplicate_conformer(water, [stretched], rthr=0.5))
    assert not any(geom.is_duplicate_conformer(water, [stretched], rthr=1e-6))
