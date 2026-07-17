# Interoperability

`Geometry` stays deliberately minimal — symbols, coordinates, charge, and
spin — so that conversions to other libraries' representations live outside
the core model. `automol.geom` provides the `Geometry`-facing conversions;
`automol.rd.mol` provides the lower-level RDKit building blocks they're
built from.

## RDKit

```python
from automol import geom

mol = geom.rdkit_mol(water)         # Geometry -> rdkit.Chem.Mol
water_rt = geom.from_rdkit_mol(mol)  # rdkit.Chem.Mol -> Geometry
```

`from_rdkit_mol` works whether or not the `Mol` already has 3D coordinates —
if it doesn't, coordinates are embedded automatically before extraction.

For lower-level RDKit work that doesn't go through `Geometry` at all,
`automol.rd.mol` operates directly on `rdkit.Chem.Mol` objects:

```python
from automol.rd import mol

water_mol = mol.from_smiles("O", with_coords=True)
mol.smiles(water_mol)         # canonical SMILES
mol.inchi(water_mol)          # InChI
mol.symbols(water_mol)        # ["O", "H", "H"]
mol.coordinates(water_mol)    # (N, 3) array; raises GeometryConversionError
                               # if the Mol has no coordinates
```

Other utilities in this module include `add_atom_numbers` (for RDKit atom
labels), `canonical_ranks` (RDKit's canonical atom ranking), and
`assign_stereochemistry`/`chiral_centers` (stereochemistry from 3D
coordinates).

## ASE

```python
atoms = geom.to_ase(water)  # Geometry -> ase.Atoms
```

`charge` and `spin` are carried over in `atoms.info`. There is currently no
`from_ase`; go through xyz or RDKit if you need the reverse direction.

## StereoMolGraph

```python
smg = geom.stereo_mol_graph(water)  # Geometry -> stereomolgraph.StereoMolGraph
```

This is what `rdkit_mol` uses internally to infer connectivity and
stereochemistry from coordinates before building the RDKit `Mol`.

## Adding a new conversion

`automol` enforces a module layering (`view` > `ident` > `geom` > `rd` >
`utils`; see the `import-linter` contract in `pyproject.toml`) where each
module may only depend on modules below it. `geom/core.py` depends on `rd`
(not the reverse) precisely because `rd` sits below `geom` in that
layering, which is why the RDKit conversions (`rdkit_mol`/`from_rdkit_mol`)
live in `geom/core.py` while the lower-level, `Geometry`-agnostic RDKit
utilities they're built from live in `rd/mol.py`.

If you're adding support for another third-party library, follow that same
split: put library-specific, `Mol`/`Atoms`/etc.-facing utilities in their
own module beneath `geom` in the layering, and put the `Geometry`-facing
conversion functions in `geom` itself (as `rdkit_mol`/`from_rdkit_mol` and
`to_ase` already do) — rather than adding a new top-level dependency to the
`Geometry` model itself.
