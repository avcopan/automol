"""Inertia and rotational analysis tests."""

import numpy as np
import pytest

from automol import Geometry, geom


@pytest.fixture
def water() -> Geometry:
    """Water geometry fixture."""
    return Geometry(
        symbols=["O", "H", "H"],
        coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        charge=0,
        spin=0,
    )


def test__inertia_tensor(water: Geometry) -> None:
    """Test that the inertia tensor is a symmetric (3, 3) matrix."""
    tensor = geom.inertia_tensor(water)
    assert tensor.shape == (3, 3)
    assert np.allclose(tensor, tensor.T)


def test__inertia_moments(water: Geometry) -> None:
    """Test that inertia moments are non-negative and match the tensor's eigenvalues."""
    moments = geom.inertia_moments(water)
    axes = geom.inertia_axes(water)
    assert moments.shape == (3,)
    assert np.all(moments >= 0)
    tensor = geom.inertia_tensor(water)
    assert np.allclose(axes.T @ tensor @ axes, np.diag(moments), atol=1e-10)


def test__inertia_axes_orthonormal(water: Geometry) -> None:
    """Test that inertia axes form an orthonormal basis."""
    axes = geom.inertia_axes(water)
    assert np.allclose(axes.T @ axes, np.eye(3))


def test__rotational_analysis_right_handed(water: Geometry) -> None:
    """Test that rotational_analysis returns a right-handed frame."""
    evals, evecs = geom.rotational_analysis(water)
    assert evals.shape == (3,)
    assert evecs.shape == (3, 3)
    assert np.linalg.det(evecs) > 0


def test__rotation_to_inertia_axes(water: Geometry) -> None:
    """Test that the returned rotation diagonalizes the inertia tensor."""
    rot = geom.rotation_to_inertia_axes(water)
    rotated = geom.core.rotate(water, rot)
    tensor = geom.inertia_tensor(rotated)
    off_diag = tensor - np.diag(np.diagonal(tensor))
    assert np.allclose(off_diag, 0, atol=1e-10)


def test__eckart_frame(water: Geometry) -> None:
    """Test that the Eckart frame centers mass and diagonalizes inertia."""
    eck = geom.eckart_frame(water)
    assert np.allclose(geom.center_of_mass(eck), 0, atol=1e-10)
    tensor = geom.inertia_tensor(eck)
    off_diag = tensor - np.diag(np.diagonal(tensor))
    assert np.allclose(off_diag, 0, atol=1e-10)


def test__eckart_frame_in_place(water: Geometry) -> None:
    """Test that eckart_frame(in_place=True) mutates and returns the same object."""
    result = geom.eckart_frame(water, in_place=True)
    assert result is water
    assert np.allclose(geom.center_of_mass(water), 0, atol=1e-10)


def test__eckart_frame_not_in_place(water: Geometry) -> None:
    """Test that eckart_frame (default) does not mutate the original geometry."""
    original = water.coordinates.copy()
    geom.eckart_frame(water)
    assert np.allclose(water.coordinates, original)
