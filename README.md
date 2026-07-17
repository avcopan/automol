# automol

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)

automol is a Python library for working with molecular geometries. It centers on a single `Geometry` model (atomic symbols, Cartesian coordinates, charge, and spin) and builds identity generation (InChI, SMILES, ...), 3D visualization, and RDKit/ASE interoperability on top of it.

See the [documentation](https://avcopan.github.io/automol/) for API reference.

## Installation

automol is built and distributed with [Pixi](https://pixi.sh). In another Pixi-managed project:

```bash
pixi add --channel avcopan automol
```

For local development, see [Contributing](CONTRIBUTING.md).

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
from automol import Algorithm, Identity

inchi = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_INCHI)
print(inchi.value)  # "InChI=1S/H2O/h1H2"

smiles = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_SMILES)
print(smiles.value)  # "O"
```

### RDKit interoperability

`automol.rd` brings RDKit functions to the project, without knowing anything about the core `Geometry` model itself:

```python
from automol import geom, rd

mol = rd.mol.from_smiles("O")
geo = geom.from_rdkit_mol(mol)
```

Conversions between `Geometry` and other cheminformatic packages (e.g., `RDKit`, `ASE`, ...) are maintained  by `automol.geom` and modules wrapping the external packages are agnostic to `automol`.

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

See the [documentation](https://avcopan.github.io/automol/) for more on geometries, identity generation, visualization, and interoperability with RDKit/ASE.

## Architecture

automol is organized as a strict layer stack (`ident` → `geom` → `rd` → `utils`), enforced by import-linter, with each core data model owned by the module that defines it — external-library bridges (like the RDKit interop above) live in consuming sub-packages rather than in the core models themselves. See [Contributing](CONTRIBUTING.md) for the full architectural rationale.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. See [Contributing](CONTRIBUTING.md) for coding standards and workflow.

## License

This project is licensed under the [MIT License](LICENSE).
