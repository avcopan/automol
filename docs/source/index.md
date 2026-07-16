# automol

automol is a Python library for working with molecular geometries. It
centers on a single `Geometry` model (atomic symbols, Cartesian
coordinates, charge, and spin) and provides:

- {doc}`geometry` — creating, reading/writing (xyz), and transforming
  `Geometry` objects, plus geometric properties like distance matrices and
  connectivity.
- {doc}`identity` — generating and comparing molecular identifiers (InChI,
  SMILES) from a `Geometry`, and back.
- {doc}`visualization` — interactive 3D viewing and static/animated
  rendering.
- {doc}`interoperability` — converting to and from RDKit, ASE, and
  StereoMolGraph representations.

New here? Start with {doc}`installation`, then {doc}`geometry`.

:::{toctree}
:maxdepth: 2
:caption: User guide

installation
geometry
identity
visualization
interoperability
:::

:::{toctree}
:maxdepth: 2
:caption: Reference

apidocs/index
:::