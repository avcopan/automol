# automol

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)

automol is a Python library for working with molecular geometries. It
centers on a single `Geometry` model (atomic symbols, Cartesian
coordinates, charge, and spin) and builds identity generation (InChI,
SMILES), 3D visualization, and RDKit/ASE interoperability on top of it.

See the [full documentation](https://avcopan.github.io/automol/) for a
complete user guide and API reference.

## Installation

automol is built and distributed with [Pixi](https://pixi.sh). In another
Pixi-managed project:

```bash
pixi add --channel avcopan automol
```

For local development, see [Contributing](CONTRIBUTING.md).

## Usage

```python
from automol import Algorithm, Geometry, Identity, View

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

# Identity
inchi = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_INCHI)
print(inchi.value)  # "InChI=1S/H2O/h1H2"

# Visualization (e.g. in a Jupyter notebook)
view = View()
view.add_geometry(water, label=True)
```

See the [full documentation](https://avcopan.github.io/automol/) for more
on geometries, identity generation, visualization, and interoperability
with RDKit/ASE.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change. See
[Contributing](CONTRIBUTING.md) for coding standards and workflow.

## License

This project is licensed under the [MIT License](LICENSE).
