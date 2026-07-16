# Molecular identity

`automol.Identity` wraps a string identifier (InChI, SMILES, ...) together
with the algorithm that produced it, so identifiers from different
algorithms are never accidentally compared or mixed up.

## Generating an identity from a `Geometry`

```python
from automol import Algorithm, Geometry, Identity

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)

inchi = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_INCHI)
smiles = Identity.from_geometry(water, algorithm=Algorithm.RDKIT_SMILES)

inchi.value       # "InChI=1S/H2O/h1H2"
inchi.algorithm   # Algorithm.RDKIT_INCHI
inchi.kind        # "stereoisomer"
```

If you already have a string identifier from elsewhere, wrap it directly
with `from_value` instead of recomputing it:

```python
inchi = Identity.from_value("InChI=1S/H2O/h1H2", algorithm=Algorithm.RDKIT_INCHI)
```

## Going back to a `Geometry`

Algorithms that support the inverse direction can reconstruct a `Geometry`
from the identifier:

```python
water_rt = inchi.geometry()
```

Calling `.geometry()` on an identity produced by an algorithm with no known
inverse raises `NotImplementedError`.

## `kind`

Every `Algorithm` is tagged with a `kind` — a category describing what sort
of identity it produces (currently `"stereoisomer"` for both built-in
algorithms). `Identity.kind` is set automatically by `from_geometry` and
`from_value`, and is validated against `algorithm.kind` on construction — an
explicit mismatch raises a `ValueError`. This lets code group or dispatch on
`kind` without hardcoding a specific algorithm.

## How algorithms are implemented

`Algorithm` is a closed `StrEnum` — each member is tagged with its `kind` at
definition time, e.g. `RDKIT_INCHI = ("rdkit inchi", "stereoisomer")`. The
behavior for each member is registered separately, via
`automol.ident.AlgorithmRegistry`, by subclassing `AlgorithmFns` and
decorating it:

```python
from automol.ident import AlgorithmFns, AlgorithmRegistry

@AlgorithmRegistry.register(Algorithm.RDKIT_INCHI)
class RDKitInChI(AlgorithmFns):
    @staticmethod
    def identity_fn(geo: Geometry) -> str:
        ...  # Geometry -> InChI

    @staticmethod
    def geometry_fn(value: str) -> Geometry:
        ...  # InChI -> Geometry
```

`geometry_fn` is optional — omit it (or fall back to `AlgorithmFns`'s
default) for an algorithm that only supports the forward direction; calling
`.geometry()` on such an identity raises `NotImplementedError`, as above.

This split — a fixed enum of *what* algorithms exist, plus a registry of
*how* each one behaves — is what `RDKitInChI` and `RDKitSMILES` in
`automol.ident` use, and is the pattern to follow when adding a new
algorithm to automol itself (which means adding a new `Algorithm` member
alongside its `AlgorithmFns` implementation, both within `automol.ident`).

Registering an algorithm twice raises
`automol.utils.exc.AlgorithmAlreadyRegisteredError`; looking up one that was
never registered raises `automol.utils.exc.UnknownAlgorithmError`.
