# Molecular identity

`automol.ident` generates string identifiers (InChI, SMILES, molecular
formulas, ...) from a `Geometry` and, where possible, reconstructs a
`Geometry` from an identifier. Each algorithm is an `automol.Algorithm`
instance; the built-in ones are `rdkit_inchi`, `rdkit_smiles`, and
`hill_formula`.

## Generating an identifier from a `Geometry`

```python
from automol import Geometry, hill_formula, rdkit_inchi, rdkit_smiles

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

rdkit_inchi.identity_fn(water)   # "InChI=1S/H2O/h1H2"
rdkit_smiles.identity_fn(water)  # "O"
hill_formula.identity_fn(water)  # "H2O"
```

`hill_formula` uses Hill order: with carbon present, C comes first, then H,
then the remaining elements alphabetically; without carbon, all elements
(including H) are alphabetical (e.g. `"ClH"` for HCl).

## Going back to a `Geometry`

Algorithms that support the inverse direction reconstruct a `Geometry` from
the identifier, with reproducible (seeded) 3D coordinates:

```python
water_rt = rdkit_inchi.geometry_fn("InChI=1S/H2O/h1H2")
```

An invalid identifier raises a `ValueError`. Calling `geometry_fn` on an
algorithm with no known inverse (such as `hill_formula`) raises
`NotImplementedError`.

## `kind`

Every algorithm is tagged with an `IdentityKind` describing what sort of
identity it produces (`"formula"`, `"isomer"`, `"stereoisomer"`, or
`"conformer"`). This lets code group or dispatch on `kind` without
hardcoding a specific algorithm.

## Parent algorithms and `other_geos`

An identity function has the signature `(geo, other_geos=None) -> str`.
`other_geos` maps previously assigned identifiers to their geometries, which
lets an algorithm reuse an existing identifier for an equivalent species.
An algorithm's `parent_algorithm` is the algorithm used to decide that
equivalence. For example, `rdkit_smiles` has `rdkit_inchi` as its parent: if
a geometry in `other_geos` has the same InChI, its (possibly non-canonical)
SMILES key is returned; otherwise RDKit's canonical SMILES is returned.

```python
other_geos = {"C(C)CCC": rdkit_smiles.geometry_fn("C(C)CCC")}
pentane = rdkit_smiles.geometry_fn("CCCCC")
rdkit_smiles.identity_fn(pentane, other_geos)  # "C(C)CCC"
```

## Registering an algorithm

Algorithms are registered with `automol.ident.AlgorithmRegistry`, so
higher-level packages can add their own without touching `automol` itself:

```python
from automol import rdkit_inchi
from automol.ident import AlgorithmRegistry, IdentityKind


def my_identity_fn(geo, other_geos=None) -> str:
    ...  # Geometry -> identifier


my_algorithm = AlgorithmRegistry.register(
    name="my algorithm",
    kind=IdentityKind.CONFORMER,
    identity_fn=my_identity_fn,
    parent_algorithm=rdkit_inchi,  # optional
)
```

`geometry_fn` (signature `(value) -> Geometry`) is optional. A function whose
signature is incompatible with these shapes is rejected with a validation
error. All registered algorithms are listed in `AlgorithmRegistry.algorithms`,
and registering a name twice raises
`automol.utils.exc.AlgorithmAlreadyRegisteredError`.
