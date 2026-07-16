# Installation

automol targets Python 3.12+ and is built and distributed with
[Pixi](https://pixi.sh).

## As a dependency

In another Pixi-managed project, add automol from the `avcopan` conda
channel:

```bash
pixi add --channel avcopan automol
```

Some functionality is optional and pulled in through the dependencies it
already declares (RDKit for identity/InChI/SMILES conversions, py3Dmol and
xyzrender for visualization) — these install automatically with the package,
so no extra flags are needed.

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
