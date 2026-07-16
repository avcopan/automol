# Geometries

The `Geometry` model and the `automol.geom` module are the core of automol:
almost everything else (identities, visualization, RDKit interoperability)
is built on top of `Geometry`.

## Creating a `Geometry`

A `Geometry` is a pydantic model with four fields: atomic `symbols`,
Cartesian `coordinates` (in Angstroms), `charge`, and `spin` (number of
unpaired electrons, i.e. `2S`):

```python
from automol import Geometry

water = Geometry(
    symbols=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]],
    charge=0,
    spin=0,
)
```

`coordinates` must have shape `(len(symbols), 3)` — a mismatch raises a
`ValueError` at construction time, since this is standard pydantic field
validation.

### Per-atom properties

`Geometry` exposes several derived, per-atom properties, backed by
`automol.utils.element`'s periodic table data:

```python
water.atom_count       # 3
water.masses           # isotopic masses, e.g. [15.9949, 1.0078, 1.0078]
water.atomic_numbers   # [8, 1, 1]
water.covalent_radii   # Pyykko covalent radii, in Angstroms
water.valences          # number of valence electrons per atom
```

## Reading and writing xyz

`Geometry` round-trips through the standard xyz format, both as instance
methods and as module-level functions in `automol.geom`:

```python
from automol import geom

xyz = water.xyz_block(comment="water")
water_rt = Geometry.from_xyz_block(xyz, charge=0, spin=0)

water.xyz_file(path="water.xyz")
water_rt = Geometry.from_xyz_file("water.xyz", charge=0, spin=0)

# equivalent module-level functions
xyz = geom.xyz_block(water, comment="water")
water_rt = geom.from_xyz_block(xyz, charge=0, spin=0)
```

`charge` and `spin` aren't part of the xyz format, so they must be supplied
explicitly when reading. A malformed or empty xyz block raises
`automol.utils.exc.XYZFormatError`.

## Molecular formula

```python
geom.hill_formula(water)  # "H2O"
```

Elements are ordered with carbon first, then hydrogen, then the rest
alphabetically (Hill order) — the standard convention regardless of whether
carbon is present.

## Geometric properties

`automol.geom` also exposes a handful of properties computed directly from
coordinates:

```python
geom.center_of_mass(water)     # mass-weighted centroid
geom.distance_matrix(water)    # pairwise atom-atom distances
geom.adjacency_matrix(water)   # binary connectivity, from covalent radii
geom.distance_keys(water)      # sorted (z1, z2, distance) descriptor
```

`adjacency_matrix` draws an edge between two atoms when their distance is
less than `sigma` times the sum of their covalent radii (`sigma=1.3` by
default). `distance_keys` produces an order-independent geometric fingerprint
useful for comparing or hashing geometries.

## Transformations

`geom.transform` provides rigid transformations that return a new `Geometry`
by default, or mutate in place with `in_place=True`:

```python
from scipy.spatial.transform import Rotation

shifted = geom.transform.translate(water, [1.0, 0.0, 0.0])
rotated = geom.transform.rotate(water, Rotation.from_euler("z", 90, degrees=True))
mirrored = geom.transform.reflect(water, normal=[0, 0, 1])
```

All three accept a `keys` argument to restrict the transformation to a
subset of atoms by index, leaving the rest untouched:

```python
geom.transform.translate(water, [1.0, 0.0, 0.0], keys=[0])  # move only atom 0
```

## Duplicate conformer detection

`is_duplicate_conformer` checks whether a `Geometry` is geometrically
identical to any geometry in a list, using
[irmsd](https://pypi.org/project/irmsd/)'s interatomic RMSD after optimal
alignment (translation, rotation, and atom matching) — so a rotated or
translated copy still counts as a match, while a genuinely different
conformer doesn't:

```python
rotated_water = geom.transform.rotate(water, Rotation.from_euler("z", 60, degrees=True))
geom.is_duplicate_conformer(water, [rotated_water])  # True

geom.is_duplicate_conformer(water, [])  # False — nothing to compare against
```

Candidates with a different atom count are never considered a match.
`rthr` (default `0.125`, matching irmsd's own default) sets the iRMSD
threshold, in Angstroms, below which two geometries count as identical —
lower it for stricter matching:

```python
geom.is_duplicate_conformer(water, [rotated_water], rthr=1e-6)  # stricter
```

## Next steps

- [Molecular identity](identity.md) — generate InChI/SMILES from a
  `Geometry`, or a `Geometry` from one.
- [Visualization](visualization.md) — view or render a `Geometry`.
- [Interoperability](interoperability.md) — convert to/from RDKit, ASE, and
  StereoMolGraph.
