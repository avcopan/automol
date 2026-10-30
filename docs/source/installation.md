# Installation

automol targets Python 3.12+ and is built and distributed with
[Pixi](https://pixi.sh).

## As a dependency

In another Pixi-managed project, add automol from the `avcopan` conda
channel:

```bash
pixi add --channel avcopan automol
```

RDKit (identity/InChI/SMILES conversions) and py3Dmol (interactive
viewing) install automatically with the package.

xyzrender, which `geom.render_svg` and `geom.render_gif` use, is only on PyPI,
so the conda package leaves it out. To render images, add it as a PyPI
dependency:

```bash
pixi add --pypi xyzrender
```

Everything else works without it; the render functions raise an `ImportError`
explaining this if it's missing. Installing automol from PyPI
(`pip install automol`) includes xyzrender.

## For development

1. Install [Pixi](https://pixi.prefix.dev/latest/installation/).
2. Fork and clone the repository.
3. From the repository root, run:

   ```bash
   pixi run init
   ```

   This sets up the `dev` environment (`pixi run -e dev ...`) with automol
   installed in editable mode, plus test, lint, and docs tooling.
4. Run the test suite to confirm the setup:

   ```bash
   pixi run -e dev pytest
   ```

See [Contributing](https://github.com/avcopan/automol/blob/main/CONTRIBUTING.md)
for coding standards and the rest of the development workflow.
