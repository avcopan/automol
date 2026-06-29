"""View functions."""

from collections.abc import Sequence

import numpy as np
import py3Dmol
from numpy.typing import ArrayLike

from . import geom
from .geom import Geometry


class View(py3Dmol.view):
    """Class for creating and displaying 3D molecular views."""

    def add_geometry(self, geo: Geometry, *, label: bool = False) -> None:
        """Add geometry to view.

        Parameters
        ----------
        geo
            Geometry.
        """
        geom.view(geo, view=self, label=label)

    def add_xyz_axes(
        self,
        *,
        scale: float = 1,
        colors: tuple[str, str, str] = ("red", "green", "blue"),
    ) -> None:
        """Add inertia axes for a geometry.

        Parameters
        ----------
        geo
            Geometry.
        """
        axes = np.eye(3)
        self.add_vectors(axes * scale, colors=colors)

    def add_vectors(
        self,
        coords: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        colors: Sequence[str] | None = None,
    ) -> None:
        """Add arrow to view.

        Parameters
        ----------
        coord
            The arrow tip coordinates.
        start_coord
            The arrow start coordinates.
        direction
            If True, coord is treated as a direction vector from start_coord.
        color
            The arrow color.
        """
        coords = np.asarray(coords, dtype=np.float64)
        colors = colors or ["black"] * len(coords)
        if len(coords) != len(colors):
            msg = f"Coordinates and colors do not match: {coords = }, {colors = }"
            raise ValueError(msg)

        for coord, color in zip(coords, colors, strict=True):
            self.add_vector(coord, start_coord, direction=direction, color=color)

    def add_vector(
        self,
        coord: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        color: str = "black",
    ) -> None:
        """Add arrow to view.

        Parameters
        ----------
        coord
            The arrow tip coordinates.
        start_coord
            The arrow start coordinates.
        direction
            If True, coord is treated as a direction vector from start_coord.
        color
            The arrow color.
        """
        if direction:
            coord = np.add(coord, start_coord)

        start = np.asarray(start_coord).tolist()
        end = np.asarray(coord).tolist()

        arrow_spec = {
            "start": {"x": start[0], "y": start[1], "z": start[2]},
            "end": {"x": end[0], "y": end[1], "z": end[2]},
            "color": color,
        }
        self.addArrow(arrow_spec)
