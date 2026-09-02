# Contributing to automol

Thank you for your interest in contributing to **automol**!

Contributions of all kinds are welcome, including bug reports, documentation improvements, and new features.

## Development workflow

To get set up:
1. Install [Pixi](https://pixi.prefix.dev/latest/installation/)
2. Fork the repository
3. Clone the repository and run `pixi run init` inside it

To contribute code, submit pull requests with clear descriptions of the changes. For larger contributions, create an issue first to propose your idea.

All tasks run through Pixi (`pixi run <task>`), defined in `pixi.toml` under `[feature.dev.tasks]`:

- `pixi run fmt` — format with Ruff
- `pixi run lint` — lint with Ruff (`--fix`)
- `pixi run types` — static type-check with `ty`
- `pixi run imports` — enforce module layering with `lint-imports` (import-linter)
- `pixi run test` — run the full pytest suite
- `pixi run pre-commit` — run all of the above via lefthook, in order (fmt → lint → types → imports → test), then check the tree is clean
- `pixi run cov-view` — open the HTML coverage report

To run a single test, invoke `pytest` directly inside the pixi env rather than through the `pixi run test` task, e.g.:

```bash
pixi run -e dev pytest tests/geom/test_core.py::test_foo
```

pytest is configured with `--doctest-modules`, so doctests in `src/` docstrings are collected and run as part of the suite. Coverage must stay ≥80% (`fail_under = 80` in `pyproject.toml`).

## Coding standards

Coding standards are largely enforced by the pre-commit hooks, which perform formatting and linting ([Ruff](https://github.com/charliermarsh/ruff)), import linting ([Lint-Imports](https://import-linter.readthedocs.io/en/stable/)), static type-checking ([Ty](https://github.com/astral-sh/ty)), and testing ([PyTest](https://docs.pytest.org/en/latest/)) with code coverage reports [CodeCov](https://docs.codecov.com/docs).

Docstrings follow the [NumPy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). Doctest examples embedded in docstrings are executed as part of the test suite (`--doctest-modules`), so keep them runnable and accurate — a docstring example that no longer works is treated as a test failure, not just stale documentation.

## Architecture

### Module layering

`pyproject.toml` defines a strict layer contract, enforced by `lint-imports` (`pixi run imports`).
Higher layers may depend on lower ones, never the reverse:

```
automol.ident   (highest)
automol.geom
automol.rd
automol.utils   (lowest)
```

`automol.rd` (RDKit interop) sits below `automol.geom`, which is itself below `automol.ident` (InChI/SMILES generation). Adding an import that violates this order will fail `pixi run imports`.

3D visualization (`automol.geom.view`) lives inside `automol.geom` rather than as its own layer, since it consumes `Geometry` directly (via `Geometry._repr_html_`) and would otherwise create a reverse dependency from `automol.geom` up to a sibling `automol.view`.

### "If you own the data, you own the interface"

Each core data model (e.g. `Geometry`) is owned by the module that defines it, and that module owns the conversion logic to/from external formats or libraries, not the bridge sub-package:

- Bridge/interop sub-packages (e.g. `automol.rd`) stay "pure" — zero knowledge of the core models. `automol.rd.mol` only deals in RDKit `Mol` objects (plus primitive types), never `Geometry`. This follows directly from the layering contract: `automol.rd` sits *below* `automol.geom`, so it cannot import `Geometry` even if it wanted to.
- The core model's own module imports the lower-layer bridge and owns the conversion, e.g. `geom/core.py` imports `automol.rd` and defines `rdkit_mol()` / `from_rdkit_mol()` to convert to/from RDKit `Mol` objects, rather than `automol.rd` knowing about `Geometry`.
- When another package in the suite (e.g. AutoStore) needs to convert an automol `Geometry`, it calls automol's own conversion function rather than reimplementing the mapping.

Conversions are standalone module-level functions, not methods on the Pydantic model, so the model itself stays free of optional/heavy dependencies.

### Current module map

- `automol.geom` — the `Geometry` core model (`core.py`), plus `properties.py` (center of mass, distance matrix/keys, adjacency matrix), `transform.py`, `comparison.py`, and `view.py` (`View`, 3D visualization, py3Dmol-based).
- `automol.rd` — RDKit bridge (`rd/mol.py`) for `Geometry` ↔ RDKit mol conversion.
- `automol.ident` — `Algorithm`/`Identity` for InChI/SMILES generation from a `Geometry`.
- `automol.utils` — shared low-level helpers: `constants.py`, `types.py`, `exc.py`, `utils/element/`.

## Conventions

automol distinguishes **modules**, **types**, and **data-valued variables** to keep scientific code concise and free of name collisions:

- **Modules** are short, singular domain nouns and act only as namespaces for algorithms, e.g. `automol.geom`, `automol.ident`. Never use a module name as a variable name.
- **Types** are singular, capitalized data models, e.g. `Geometry`.
- **Variables** holding instances use short abbreviations distinct from the module name, e.g. a `Geometry` instance is `geo` (not `geom`, which is the module).
- **Algorithms are module-level functions**, not instance methods, e.g. `center_of_mass(geo: Geometry) -> FloatArray` in `automol.geom`.

```python
from automol import geom, Geometry

geo = Geometry(["O", "H", "H"], coordinates)
com = geom.center_of_mass(geo)
```

This holds across sub-package boundaries too — the RDKit bridge follows the same pattern of module-level conversion functions rather than methods on `Geometry` or `Mol`:

```python
from automol import geom, rd

mol = rd.mol.from_smiles("O")
geo = geom.from_rdkit_mol(mol)
```

## Questions

If you have questions about contributing or design decisions, feel free to open an issue for discussion.
