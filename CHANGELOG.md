# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
### Added
- `geom.transform.transition()` for determining the transition-state geometry between two geometries via `StereoCondensedReactionGraph`.

## [0.0.19] - 2026-07-17
### Added
- `Algorithm.IRMSD` for tagging conformer-group identities (unregistered algorithm; built directly via `Identity.from_value` rather than `from_geometry`).
- `geom.comparison.is_duplicate_conformer()` for iRMSD-based conformer matching.
- `geom.adjacency_matrix(..., flood_fill=True)` option for connectivity-based flood filling.
- `rd.mol.set_coordinates()` for replacing an RDKit mol's conformer coordinates.
- `geom.inertia`, `geom.internal`, `geom.vibration` modules (experimental): moments of inertia, internal coordinates, and vibrational analysis.
- `docs/source/*` pages (geometry, identity, interoperability, visualization, installation) and expanded README.

### Changed
- `automol.element` -> `automol.utils.element` to fit module layering.
- `automol.view` -> `automol.geom.view` since it consumes `Geometry` directly.
- `tests/test_geom.py` -> `tests/{test_core,test_properties,test_transform,test_comparison}.py` + `tests/conftest.py` to organize growing test suite.

### Removed
- `automol.geom.canon`, `automol.geoms`, `automol.graph` (including `graph.ts`) — superseded by current `geom`/`ident` design.

## [0.0.18] - 2026-07-02
### Added

### Changed
- `Geometry.canonical_form(self, *, in_place=True)` -> `.canonical_form(self, *, delta ...)` to support method chaining and discourage in-place operations on SQLModel subclasses.
- `_float_array_validator(...)` returns `np.array(obj, dtype...)` instead of `np.asarray(obj, dtype...)` due to instantiation concerns when hashing.

### Fixed
- Bug with `... for targets in nx.all_pairs_shortest_path...` exposed when operating on purely cyclic molecules.
- Premature raise on `Geometry.validate_coordinates_shape(...)` model validator exposed when validating SQLModel subclasses.
- Premature hash setting on `Geometry.set_hash()` model validator exposed when validating SQLModel subclasses.
- Instance building on `Geometry.canonical_form()` exposed when canonicalizing SQLModel subclasses.
- `test__deterministic_canonical_order(...)` to reflect updates.

### Removed
- `Geometry.sort()`.

## [0.0.17] - 2026-06-29
### Added
- `elements`, `constants`, and `ident` from `automatics` (package discontinued).
- `canonical_frame` to `geom` module for expanding `eckart_frame` logic to include sign choices.
- `canonical_sorting` to `geom` module for initial implementation of standardized atom sorting.

### Changed
- `geom.py` -> `geom/*` to better organize growing codebase.

### Fixed
- `tests` to reflect changes in update.

### Removed
- `is_similar` due to canonical framing and sorting.

## [0.0.16] - 2026-06-18
### Added
- `automatics.geom` module exports within `automol.geom` to avoid namespace clash.
- `harmonic_zpv` (harmonic zero point vibrational energy) method.
- `xyzrender` as a developer / optional dependency.

### Changed
- Propyl oxirane test fixtures to read objects from data files.
- Bump `automatics` to v0.0.6.

## [0.0.15] - 2026-06-17
### Added
- `geom.vibrational_analysis()` and corresponding functions to calculate frequencies from a `Geometry` and its Hessian.

### Changed
- Unit conversions import from `automatics`.
- `kabsch()` and `is_similar()` relocated from `geom` to `geoms`.

### Fixed
- Bump `automatics` to v0.0.5.
- Layering to incorporate new `geoms` module.

### Removed
- Minor comments in src files.

## [0.0.14] - 2026-06-12
### Added
- Dependency on automatics (0.0.4).

### Changed
- Update tests to reflect refactors.

### Removed
- Geometry, Identity, and View relocated to automatics for consistent source of truth in autosuite.
- Miscellaneous utility scripts pertaining to Geometry, Identity, and View.


## [0.0.13] - 2026-06-03
- Implemented Identity class with boilerplate for handling conversions between Geometry and chemical identifiers such as InChI / SMILES.
- Converted qcdata to an optional dependency with conversion methods placed in geom.py.
- Implemented a decorator for optional qc data dependency.
- Dropped qccompute from pixi.toml pypi-dependency list.

## [0.0.12] - 2026-05-20
- geom.is_similar() checks InChI first, no longer considers moment of inertia deviation, and ensures that geometry symbols are identically ordered between geo1 and geo2 (kabsch implentation is order dependent).
- Added nvalence, covalent radius, and group to elements-data.
- geom.determine_neighbors() as a first attempt at defining connectivity from geometries.

## [0.0.11] - 2026-05-04
- Added missing pyparsing dependency

## [0.0.10] - 2026-04-30
- Renames functions and arguments for clarity and consistency

## [0.0.9] - 2026-04-18
- Overhaul graph API with better design and better typing as Graph[Atom, Bond]
- Implement graph.ts submodule with brute-force reaction mapping algorithm

## [0.0.8] - 2026-04-16
- Added view submodule for building view objects
- Added geometry functions (translation, rotation, reflection, dihedral angles, etc.)
- Added graph submodule with conversion to/from RDKit Mol and SMILES/InChI

## [0.0.7] - 2026-04-08
- Added inertia moments, kabsch alignment, and center of mass algebraic methods to geom.py
- Added similarity analysis to geom.py (mirroring first two steps of prism_pruner)
- Added distance setting to geom.py

## [0.0.6] - 2026-04-01

## [0.0.5] - 2026-01-29
### Added
- Geometry hash function to root namespace

## [0.0.4] - 2026-01-29
### Changed
- Renamed geometry hash function to `geometry_hash()` to avoid shadowing built-in `hash()`

## [0.0.3] - 2026-01-28
### Added
- Geometry hash function

## [0.0.2] - 2026-01-28
### Fixed
- Fix Geometry.coordinate type annotation

## [0.0.1] - 2026-01-26
### Added
- Generate Geometry from SMILES
- Calculate Geometry center of mass
