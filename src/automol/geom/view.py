"""View functions."""

from pathlib import Path

import py3Dmol
import xyzrender

from .core import Geometry, xyz_file


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

    tmp_file = Path.cwd() / ".tmp.xyz"
    xyz_file(geo, path=tmp_file)
    mol = xyzrender.load(tmp_file)

    tmp_file.unlink()
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

    tmp_file = Path.cwd() / ".tmp.xyz"
    xyz_file(geo, path=tmp_file)
    mol = xyzrender.load(tmp_file)

    tmp_file.unlink()
    return xyzrender.render_gif(
        mol, config=config, hy=include_h, output=out, gif_rot=rotation_axis
    )
