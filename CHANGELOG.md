# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.0.26] - 2026-09-30
### Added
- `rd.mol.from_connectivity(symbols, bonds, *, charge, spin, coords)` — builds a sanitized RDKit `Mol` from connectivity alone, perceiving bond orders, formal charges, and radical electrons that exactly match the requested charge and spin via an integer linear program (`scipy.optimize.milp`). Raises `NotImplementedError` for metals.
- `utils.element.electronegativity()` (Pauling) and `utils.element.is_metal()`, backed by new `electronegativity` / `metal` fields in the generated element data.
- `Geometry` validation: symbols are checked against the periodic table and canonicalized (`"cl"` → `"Cl"`), and `spin` must be non-negative and consistent with the electron count. Cross-field checks now also run on assignment, and a failed assignment leaves the geometry unchanged.
- Value-based `Geometry.__eq__`. Previously, `==` raised `ValueError` on the coordinate arrays.
- `rd.mol.max_bond_count()`, the maximum number of bonds to an atom allowing formal charges of ±1.
- `seed` parameter on `rd.mol.add_coordinates()` (default `rd.mol.EMBED_SEED`; `None` for random).
- `geom.translate`, `geom.rotate`, `geom.reflect`, `geom.view`, `geom.render_svg`, and `geom.render_gif` exported from `automol.geom`.
- `from_xyz_block` accepts atomic numbers and lowercase symbols as atom labels, and floats with a leading dot (`.74`).
- A `data` pixi environment with `mendeleev` and `sqlalchemy` for the `elements-data` task.

### Fixed
- `adjacency_matrix(..., flood_fill=True, enforce_valence=True)` no longer loops forever when valence caps prevent fragments from connecting.
- `rdkit_smiles.identity_fn` no longer raises `StopIteration` when no `other_geos` entry matches.
- `hill_formula` follows strict Hill order: without carbon, all elements (including H) are alphabetical (e.g. `ClH`).
- `Geometry.relabel_atoms` rejects indices that aren't a permutation and keeps subclass fields.
- `from_xyz_block` checks the atom count line, and rejects multi-frame blocks and fused tokens such as `H1 0 0 0` with a clear `XYZFormatError`.
- Invalid SMILES/InChI/xyz input to `rd.mol.from_smiles` / `from_inchi` / `from_xyz_block` and the `geometry_fn`s raises `ValueError` instead of an RDKit `ArgumentError`. Failed InChI generation raises instead of returning `""`.
- `rd.mol.add_coordinates` raises `GeometryConversionError` when embedding fails, after retrying with random initial coordinates.
- `geom.reflect` rejects a zero normal vector instead of returning NaN coordinates.
- `View` labels and styles apply only to the most recently added geometry.
- `center_of_mass` / `inertia_tensor` handle empty geometries, and `bonds` / `angles` / `dihedrals` return correctly shaped empty arrays.
- `Element.group` is typed `int | None`. It is `None` for lanthanides and actinides.

### Changed
- **Breaking:** The `automol.geom.core`, `automol.geom.analysis`, and `automol.geom.io` submodules are merged into a single `automol.geom` module. Import from `automol.geom` instead (e.g. `geom.reflect`, not `geom.core.reflect`). The public names are unchanged.
- `automol.utils.element` is now a single module (`utils/element.py`), so `automol.utils.element.core` no longer exists. The public `utils.element.*` functions are unchanged.
- `rd/mol.py` is reorganized into sections (conversion, accessors, coordinates, stereochemistry, Lewis structure perception). There are no API changes.
- **Documentation:** Docstrings use Google style, validated by Ruff and rendered by Sphinx through Napoleon. Types come from annotations and are no longer repeated in `Returns:` or `Attributes:`, and empty `Returns: None` sections are omitted. Module docstrings and comments are clearer and shorter.
- `geom.rdkit_mol` no longer depends on `stereomolgraph`: connectivity comes from `adjacency_matrix`, the Lewis structure from `rd.mol.from_connectivity`, and stereochemistry from the 3D coordinates. Radicals, charged species, and closed-shell singlet carbenes are now represented faithfully.
- **Breaking:** `geom.dihedrals` returns rows `[z1, z2, z3, z4, phi]` keyed by atomic numbers, oriented canonically and sorted, consistent with `bonds` and `angles`. Previously, rows held atom indices.
- `adjacency_matrix(enforce_valence=True)` caps bonds at `rd.mol.max_bond_count` instead of the valence-electron count, and keeps the shortest bonds (relative to covalent radii). The result no longer depends on atom order.
- `angles` and `dihedrals` are built from neighbor lists instead of dense O(N³)/O(N⁴) tensors, and `inertia_tensor` is vectorized.
- `rd.mol.from_smiles` / `from_inchi` (and so `geometry_fn`) embed coordinates with a fixed seed, so the results are reproducible.
- `Algorithm` validates that `identity_fn` accepts `(geo, other_geos)` and `geometry_fn` accepts `(value)`.
- `xyz_file` writes a trailing newline. `xyz_block` rejects multi-line comments.
- `rd.mol.from_inchi` sanitizes the parsed molecule.
- The `elements-data` task moved to the new `data` environment.

### Removed
- `AlgorithmRegistry.get()`, `.all_algorithms()`, `.algorithms_for_kind()` — unused lookup helpers; algorithms are now accessed via their module-level `Algorithm` instances (e.g. `rdkit_inchi`, `rdkit_smiles`, `hill_formula`) instead of by name.
- `geom.inertia_moments()`, `inertia_axes()`, `rotational_analysis()`, `rotation_to_inertia_axes()`, and `eckart_frame()`.
- `geom.mass_weight_vector()`, `translational_normal_modes()`, `rotational_normal_modes()`, `normal_mode_projection()`, `vibrational_analysis()`, and `harmonic_zpv()`.
- `geom.bond_graph()`, `orbit_classes()`, `kabsch_align()`, `hungarian_correspondence()`, and `assignment_rmsd()`.
- `geom.stereo_mol_graph()`, `from_stereo_mol_graph()`, `set_bond()`, and `transition()`.
- `pynauty` and `stereomolgraph` dependencies.
- `Algorithm.deterministic` and the `deterministic` parameter of `AlgorithmRegistry.register(...)`.
- Unused `HashGenerationError`, `UnknownAlgorithmError`, and `XTBError` exceptions, and the unused `geom.analysis.RMSD_THRESHOLD` constant.
- Public internals `geom.FLOOD_FILL_STEP`, the xyz grammar constants (`SYMBOL`, `FLOAT`, `XYZ_LINE`), and `utils.element.ELEMENT_BY_NUMBER` / `ELEMENT_BY_SYMBOL` / `PERIOD_SHELL_CAPACITY` are now private.

## [0.0.25] - 2026-09-19
### Added
- `Algorithm` `BaseModel` representing a registered algorithm as a standalone instance (`name`, `kind`, `identity_fn`, `geometry_fn`, `parent_algorithm`, `deterministic`), replacing the `AlgorithmDef` dataclass / `AlgorithmFns` ABC pair.
- `IdentityProtocol` / `GeometryProtocol` (`@runtime_checkable` `Protocol`s) describing the `identity_fn` / `geometry_fn` callable shapes, replacing the `Callable[...]` type aliases used by `AlgorithmDef`.
- `parent_algorithm` field on `Algorithm` / parameter on `AlgorithmRegistry.register(...)` so a non-canonical algorithm (e.g. `rdkit_smiles`, `hill_formula`) can disambiguate `other_geos` via a canonical parent (e.g. `rdkit_inchi`); `rdkit_smiles`'s `identity_fn` now returns the original (possibly non-canonical) SMILES from `other_geos` when its InChI matches, instead of always returning RDKit's canonical form.
- `deterministic` field on `Algorithm` / parameter on `AlgorithmRegistry.register(...)` (default `True`) flagging whether an algorithm produces deterministic strings; `rdkit_smiles` and `hill_formula` are registered as non-deterministic.
- Module-level `rdkit_inchi`, `rdkit_smiles`, `hill_formula` `Algorithm` instances, exported from the top-level `automol` namespace and called directly (`rdkit_inchi.identity_fn(...)`, `.geometry_fn(...)`) instead of through i`Identity`.
- `OTHER_GEOS` type alias (`Mapping[str, Geometry] | None`) for the `other_geos` parameter.

### Changed
- `AlgorithmRegistry.register(...)` is now a plain classmethod that registers an `Algorithm` instance directly (`AlgorithmRegistry.register(name=..., kind=..., identity_fn=..., geometry_fn=...)`), instead of a decorator applied to an `AlgorithmFns` subclass.
- `AlgorithmRegistry` stores algorithms in a public `algorithms: ClassVar[list[Algorithm]]` instead of a private `_algorithms: ClassVar[dict[str, AlgorithmDef]]`.

### Removed
- `Identity` `BaseModel` (`.from_geometry()`, `.from_value()`, `.geometry()`, and the `kind`/`algorithm` consistency validator) — replaced by calling `identity_fn` / `geometry_fn` directly on the registered `Algorithm` instances.
- `AlgorithmDef` dataclass and `AlgorithmFns` ABC — superseded by the `Algorithm` model and `IdentityProtocol` / `GeometryProtocol`.
- `AlgorithmRegistry.register_def()` — folded into `AlgorithmRegistry.register()`.
- `RDKIT_INCHI`, `RDKIT_SMILES`, `HILL_FORMULA` string constants — replaced by the `rdkit_inchi`, `rdkit_smiles`, `hill_formula` `Algorithm` instances.

## [0.0.24] - 2026-09-11
### Added
- `IdentityKind` `StrEnum` for categorizing identity types (`FORMULA`, `STEREOISOMER`, `CONFORMER`, `ISOMER`).
- `AlgorithmDef`, `AlgorithmFns`, `AlgorithmRegistry` exported from top-level `automol` namespace.
- `other_geos` parameter type (`dict[str, Geometry] | None`) in `AlgorithmFns.identity_fn` to support named reference geometries for conformer identity generation.

### Changed
- `Algorithm` `StrEnum` removed; algorithms are now plain string identifiers, so higher-level packages can register their own via `AlgorithmRegistry.register(algorithm, kind)` without modifying `automol.ident`. Built-in algorithms are exposed as module-level constants (`RDKIT_INCHI`, `RDKIT_SMILES`, `HILL_FORMULA`) instead of enum members.


## [0.0.23] - 2026-09-01
### Removed
- `geom.is_duplicate_conformer()` and `irmsd` dependency. (Broken `numpy` reference in solved `irmsd` version).
- `Algorithm.IRMSD` conformer-group identity algorithm.

### Fixed
- Dependencies listed in `pixi.toml` instead of `pyproject.toml`

## [0.0.22] - 2026-08-28
### Added
- `Geometry.relabel_atoms()` for reordering atoms by index.
- `geom.analysis.bond_graph()` / `orbit_classes()` for pynauty-based graph construction and automorphism-orbit detection.
- `geom.analysis.kabsch_align()`, `hungarian_correspondence()`, `assignment_rmsd()` for atom-correspondence-aware structural alignment and RMSD.
- `Algorithm.HILL_FORMULA` / `HillFormula` for Hill-ordered molecular formula identity via the `ident` registry.
- `pynauty` dependency.

### Changed
- `geom.{comparison,inertia,internal,properties,transform}` consolidated into `geom.analysis` (distance/inertia/vibration/comparison/graph analysis); `geom.view` renamed to `geom.io` and merged with xyz block/file I/O.
- `geom.hill_formula()` -> `Algorithm.HILL_FORMULA` / `HillFormula`, matching the pattern used by InChI/SMILES/SMG_HASH.
- `geom.transform.{translate,reflect,rotate,transition}` and `geom.internal.set_bond()` moved to `geom.core`.
- xyz parsing (`from_xyz_block`) reimplemented without `pyparsing`.
- `.gitignore`: drop `experimental`, add `.scratch`.

### Removed
- `automol.view` top-level re-export (use `automol.geom.io` / `automol.View`).

## [0.0.21] - 2026-07-26
### Added
- `geom.transform.transition()` for determining the transition-state geometry between two geometries via `StereoCondensedReactionGraph`.
- `Algorithm.SMG_HASH` / `StereoMolGraphHash` for enantiomer-invariant conformer identity via `StereoMolGraph` hashing.

### Changed
- `geom.internal.set_distance()` -> `set_bond()` for clarity.
- `geom.comparison.is_duplicate_conformer()` now returns a list the same length as `geos`, appending `False` for symbol-count mismatches instead of skipping them.


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
