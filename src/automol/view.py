"""View functions."""

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import py3Dmol
import xyzrender
from numpy.typing import ArrayLike

from .geom import Geometry, xyz_file


class View(py3Dmol.view):
    """Class for creating and displaying 3D molecular views."""

    def add_geometry(self, geo: Geometry, *, label: bool = False) -> None:
        """Add geometry to view.

        Parameters
        ----------
        geo
            Geometry.
        """
        view(geo, view=self, label=label)

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


# Visualization
def view(
    geo: Geometry, *, view: py3Dmol.view | None = None, label: bool = False
) -> py3Dmol.view:
    """View a geometry with py3Dmol.

    Parameters
    ----------
    geo
        Geometry.
    view
        py3Dmol view.
    label
        Whether to add atom labels to the view.

    Returns
    -------
        py3Dmol view.
    """
    view = py3Dmol.view(width=400, height=400) if view is None else view
    xyz_str = geo.xyz_block()
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    if label:
        for key in range(len(geo.symbols)):
            view.addLabel(
                key,
                {
                    "backgroundOpacity": 0.0,
                    "fontColor": "black",
                    "alignment": "center",
                    "inFront": True,
                },
                {"index": key},
            )
    return view


def render_svg(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: str | xyzrender.RenderConfig = "default",
    include_h: bool = True,
) -> xyzrender.SVGResult:
    """Render geometry in .svg format.

    Results display inlay automatically.

    Parameters
    ----------
    geo
        Geometry.
    out
        Output path for rendered image.
    config
        xyzrender RenderConfig settings.
    include_h
        If True, include hydrogen atoms in render.

    Returns
    -------
    SVGResult
    """
    out = Path(out).with_suffix(".svg") if out else out

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "geometry.xyz"
        xyz_file(geo, path=tmp_file)
        mol = xyzrender.load(tmp_file)
        return xyzrender.render(mol, config=config, hy=include_h, output=out)


def render_gif(
    geo: Geometry,
    *,
    out: str | Path | None = None,
    config: str | xyzrender.RenderConfig = "default",
    include_h: bool = True,
    rotation_axis: str = "x",
) -> xyzrender.GIFResult:
    """Render geometry rotating about an axis in .gif format.

    Results display inlay automatically.

    Parameters
    ----------
    geo
        Geometry.
    out
        Output path for rendered gif.
    config
        xyzrender RenderConfig settings.
    include_h
        If True, include hydrogen atoms in render.
    rotation_axis
        Axis to rotate about in animation.

    Returns
    -------
    GIFResult
    """
    out = Path(out).with_suffix(".gif") if out else out

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "geometry.xyz"
        xyz_file(geo, path=tmp_file)
        mol = xyzrender.load(tmp_file)
        return xyzrender.render_gif(
            mol, config=config, hy=include_h, output=out, gif_rot=rotation_axis
        )
