"""Coordinate I/O and 3D visualization of a geometry."""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import py3Dmol
import pyparsing as pp
import xyzrender
from numpy.typing import ArrayLike
from pyparsing import pyparsing_common as ppc

from ..utils.exc import XYZFormatError

if TYPE_CHECKING:
    from .core import Geometry

CHAR = pp.Char(pp.alphas)
SYMBOL = pp.Combine(CHAR + pp.Opt(CHAR))
XYZ_LINE = SYMBOL + pp.Group(ppc.fnumber * 3) + pp.Suppress(... + pp.LineEnd())


def xyz_block(geo: "Geometry", *, comment: str | None = None) -> str:
    """Return Geometry as a formatted xyz block with optional comment.

    Defaults to a comment reporting the charge and spin, e.g. "Geometry(q=0, s=0)".
    """
    if comment is None:
        comment = f"Geometry(q={geo.charge}, s={geo.spin})"
    lines = [str(geo.atom_count), comment]
    for sym, (x, y, z) in zip(geo.symbols, geo.coordinates, strict=True):
        lines.append(f"{sym:<4} {x:12.8f} {y:12.8f} {z:12.8f}")

    return "\n".join(lines)


def from_xyz_block(xyz_block: str, *, charge: int, spin: int) -> "Geometry":
    """Instantiate Geometry from a formatted xyz block."""
    from .core import Geometry  # noqa: PLC0415

    lines = xyz_block.strip().splitlines()[2:]

    if not lines:
        msg = "The provided xyz block is empty."
        raise XYZFormatError(msg)

    try:
        symbs, coords = zip(
            *[XYZ_LINE.parse_string(line).as_list() for line in lines], strict=True
        )
    except pp.ParseException as exc:
        msg = f"Failed to parse xyz line: {exc.line!r}"
        raise XYZFormatError(msg) from exc

    return Geometry(
        symbols=list(symbs), coordinates=np.array(coords), charge=charge, spin=spin
    )


def xyz_file(geo: "Geometry", *, path: str | Path, comment: str | None = None) -> None:
    """Write a Geometry to a formatted xyz file.

    Defaults to a comment reporting the charge and spin, e.g. "Geometry(q=0, s=0)".
    """
    Path(path).write_text(xyz_block(geo, comment=comment))


def from_xyz_file(path: str | Path, *, charge: int, spin: int) -> "Geometry":
    """Instantiate Geometry from a formatted xyz file."""
    return from_xyz_block(Path(path).read_text(), charge=charge, spin=spin)


class View(py3Dmol.view):
    """Class for creating and displaying 3D molecular views."""

    def add_geometry(self, geo: "Geometry", *, label: bool = False) -> None:
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
    geo: "Geometry", *, view: py3Dmol.view | None = None, label: bool = False
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
    geo: "Geometry",
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
    geo: "Geometry",
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
