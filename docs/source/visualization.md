# Visualization

`automol.view` builds on [py3Dmol](https://3dmol.csb.pitt.edu/) for
interactive 3D viewing (e.g. in a Jupyter notebook) and on
[xyzrender](https://pypi.org/project/xyzrender/) for static image and
animation output.

## Interactive viewing

`automol.View` is a `py3Dmol.view` subclass with convenience methods for
adding `Geometry` objects and annotations:

```python
from automol import Geometry, View

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

v = View()
v.add_geometry(water, label=True)  # label=True adds atom-index labels
v.show()
```

In a notebook, the view renders inline; `v.show()` (inherited from
`py3Dmol.view`) is only needed outside of that auto-display context.

For a one-off view without constructing a `View` yourself:

```python
from automol import view

v = view.view(water, label=True)
```

### Annotating with vectors

`add_xyz_axes` draws the standard x/y/z axes (useful for checking
orientation after a rotation), and `add_vector`/`add_vectors` draw arbitrary
arrows — for example, an inertial axis or a bond dipole:

```python
v.add_xyz_axes(scale=2.0)
v.add_vector([1.0, 0.0, 0.0], start_coord=[0.0, 0.0, 0.0], color="purple")
v.add_vectors(
    [[1, 0, 0], [0, 1, 0]],
    colors=["red", "green"],
)
```

`direction=True` treats the vector as an offset from `start_coord` rather
than an absolute endpoint.

## Rendering to files

`render_svg` and `render_gif` produce a static or rotating image without
needing a browser or notebook — useful for scripts, reports, or CI
artifacts:

```python
from automol import view

view.render_svg(water, out="water.svg")
view.render_gif(water, out="water.gif", rotation_axis="y")
```

Both accept `config` (an `xyzrender` `RenderConfig`, or `"default"`) and
`include_h` to control whether hydrogens are drawn. `out` is optional —
omit it to get the rendered result back without writing a file; the result
also displays inline automatically in a notebook.
