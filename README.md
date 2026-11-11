# automol

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)

> [!WARNING]
> **automol is deprecated and no longer maintained.** Version 0.0.32 is the last release.
> Use [autogeom](https://github.com/avcopan/autogeom) instead, which continues the `Geometry` code under a new name.
> The identity framework (`Algorithm`, `AlgorithmRegistry`, `IdentityKind`) moved to [autostorage](https://github.com/avcopan/autostorage).
> See [Moving to autogeom](#moving-to-autogeom) below.

automol is a Python library for working with molecular geometries. It centers on a single `Geometry` model (atomic symbols, Cartesian coordinates, charge, and spin) and builds identity generation (InChI, SMILES, ...), 3D visualization, and RDKit interoperability on top of it.

See the [documentation](https://avcopan.github.io/automol/) for the user guide and API reference.

## Moving to autogeom

The name `automol` clashes with the `automol` module of the older AutoMech codes, so the two could not be installed in one environment.
The code now lives on under names that don't clash:

- **[autogeom](https://github.com/avcopan/autogeom)** holds the `Geometry` model and everything that operates on it ([documentation](https://avcopan.github.io/autogeom/)).
- **[autostorage](https://github.com/avcopan/autostorage)** holds the identity framework, which only it used.

Install autogeom with `pip install autogeom`, or with `pixi add autogeom` from the [`avcopan/automech`](https://prefix.dev/channels/avcopan/automech) channel.

| automol 0.0.31 | now |
|---|---|
| `automol.Geometry`, `automol.View` | `autogeom.Geometry`, `autogeom.View` |
| `automol.geom.<function>(geo)` | `autogeom.<function>(geo)` (the `geom` module is gone) |
| `automol.rd.mol.<function>` | `autogeom.rd.<function>` |
| `automol.utils.element`, `automol.utils.types`, `automol.utils.exc` | `autogeom.element`, `autogeom.types`, `autogeom.exc` |
| `automol.Algorithm`, `automol.AlgorithmRegistry`, `automol.IdentityKind` | `autostorage.IdentityAlgorithm`, `autostorage.IdentityAlgorithmRegistry`, `autostorage.IdentityKind` |
| `automol.rdkit_inchi`, `automol.rdkit_smiles`, `automol.hill_formula` | `autostorage.rdkit_inchi`, `autostorage.rdkit_smiles`, `autostorage.hill_formula` |

The changelogs of [autogeom](https://github.com/avcopan/autogeom/blob/main/CHANGELOG.md) and [autostorage](https://github.com/avcopan/autostorage/blob/main/CHANGELOG.md) list every change.
The rest of this page describes automol 0.0.32 as it is.

## Installation

automol requires Python 3.13 or later.
With [Pixi](https://pixi.sh), add it from the [`avcopan/automech`](https://prefix.dev/channels/avcopan/automech) channel on prefix.dev:

```bash
pixi workspace channel add https://repo.prefix.dev/avcopan/automech
pixi add automol
pixi add --pypi xyzrender  # optional: only on PyPI, needed for geom.render_svg/render_gif
```

Or install it from PyPI, which includes xyzrender:

```bash
pip install automol  # or: uv add automol
```

For local development, see [Contributing](https://github.com/avcopan/automol/blob/main/CONTRIBUTING.md).

## Usage

### Geometry

Every feature builds on the single `Geometry` model — atomic symbols, Cartesian coordinates, charge, and spin:

```python
from automol import geom, Geometry

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

com = geom.center_of_mass(water)
```

### Identity generation (InChI, SMILES)

```python
from automol import hill_formula, rdkit_inchi, rdkit_smiles

rdkit_inchi.identity_fn(water)  # "InChI=1S/H2O/h1H2"
rdkit_smiles.identity_fn(water)  # "O"
hill_formula.identity_fn(water)  # "H2O"

water_rt = rdkit_inchi.geometry_fn("InChI=1S/H2O/h1H2")  # back to a Geometry
```

### RDKit interoperability

`automol.rd` brings RDKit functions to the project, without knowing anything about the core `Geometry` model itself:

```python
from automol import geom, rd

mol = rd.mol.from_smiles("O")
geo = geom.from_rdkit_mol(mol)
```

Conversions between `Geometry` and other cheminformatic packages (e.g., `RDKit`) are maintained by `automol.geom` and modules wrapping the external packages are agnostic to `automol`.

### Visualization

```python
from automol import View

# Visualization (e.g. in a Jupyter notebook)
view = View()
view.add_geometry(water, label=True)
```

`Geometry` also supports inlay rendering, providing a `View()` and xyz-formatted block:

```python
water
# Shows a 3D rendering + xyz-formatted block.
```

See the [documentation](https://avcopan.github.io/automol/) for more on geometries, identity generation, visualization, and interoperability with RDKit.

## Architecture

automol is organized as a strict layer stack (`ident` → `geom` → `rd` → `utils`), enforced by import-linter, with each core data model owned by the module that defines it — external-library bridges (like the RDKit interop above) live in consuming sub-packages rather than in the core models themselves. See [Contributing](https://github.com/avcopan/automol/blob/main/CONTRIBUTING.md) for the full architectural rationale.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. See [Contributing](https://github.com/avcopan/automol/blob/main/CONTRIBUTING.md) for coding standards and workflow.

## License

This project is licensed under the [MIT License](https://github.com/avcopan/automol/blob/main/LICENSE).

## Credits

This package was created with [copier](https://copier.readthedocs.io/) and the [avcopan/copier-pixi-python](https://github.com/avcopan/copier-pixi-python) project template.
