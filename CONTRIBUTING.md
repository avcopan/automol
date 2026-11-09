# Contributing to automol

Contributions are welcome: bug reports, documentation, and code.
For larger changes, please open an issue first to discuss the idea.

## Setup

1. Install [Pixi](https://pixi.prefix.dev/latest/installation/).
2. Fork and clone the repository.
3. In the clone, install the environments and the Git hooks:
   ```bash
   pixi install --all
   pixi run lefthook install
   ```

The project follows the [copier-pixi-python](https://github.com/avcopan/copier-pixi-python) template, whose README explains the tooling in more detail.

## Workflow

| task | what it does |
|---|---|
| `pixi run test` | Run the tests, including the doctests in `src/` |
| `pixi run test-lowest` | Run the tests with the lowest dependency versions allowed by `pyproject.toml` |
| `pixi run pre-commit` | Run every check that runs on commit: format, lint, types, import contracts, both test runs, and a clean-tree check |
| `pixi run pre-commit-check` | The same, without the tests |
| `pixi run docs-preview` | Build and preview the documentation site |
| `pixi run add <pkg>` | Add a run dependency to `pyproject.toml` |

To run a single test, call pytest in the `test` environment:
```bash
pixi run -e test pytest tests/test_geom.py::test__center_of_mass
```

Coverage must stay at or above 80%.
Docstrings use the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), with types taken from the annotations rather than repeated in `Args:` and `Returns:`.
Docstring examples run as doctests, so keep them working.

## Design

### Modules, types, and variables

automol keeps modules, types, and data-valued variables distinct, so scientific code stays short and free of name collisions:

- **Modules** are short, singular nouns that act as namespaces for algorithms, e.g. `automol.geom`. Never use a module name as a variable name.
- **Types** are singular, capitalized data models, e.g. `Geometry`.
- **Variables** holding instances use abbreviations distinct from the module, e.g. `geo` for a `Geometry` (not `geom`).
- **Algorithms are module-level functions**, not methods, e.g. `geom.center_of_mass(geo)`.

```python
from automol import geom, rd

geo = geom.from_rdkit_mol(rd.mol.from_smiles("O"))
com = geom.center_of_mass(geo)
```

### Layers

Modules are layered, and a module may only import from the layers below it.
`pixi run imports` enforces this with the contract in `pyproject.toml`:

```
automol.ident   (highest)
automol.geom
automol.rd
automol.utils   (lowest)
```

### Conversions

Each conversion between an automol model and an external library or format has exactly one home in automol.
Bridge modules know only their external library: `automol.rd` deals only in RDKit `Mol` objects and never imports automol's models.
The conversions themselves, such as `geom.rdkit_mol()` and `geom.from_rdkit_mol()`, live above the bridge, in `automol.geom`, which the layers allow.
Other packages, such as autostorage, call these conversions rather than reimplementing them.

Keeping conversions in `automol.geom` works for libraries automol depends on anyway, like RDKit.
For a library that is heavy or optional, either import it lazily inside the function that needs it, as `geom.render_svg` does with xyzrender, or put its conversions in a dedicated adapter module above both `automol.geom` and the bridge, so that `automol.geom` doesn't depend on it.
