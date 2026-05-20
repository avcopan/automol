# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
