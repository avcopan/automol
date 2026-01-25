# Contributing to automol

Thank you for your interest in contributing to **automol**!
Contributions of all kinds are welcome, including bug reports,
documentation improvements, and new features.

This document outlines the basic development workflow and coding
conventions used in the project.

------------------------------------------------------------------------

## Development workflow

1.  Fork the repository and create a feature branch.
2.  Make your changes with clear, focused commits.
3.  Ensure all tests pass and new functionality is covered by tests.
4.  Open a pull request with a clear description of the changes.

------------------------------------------------------------------------

## Coding standards

-   Follow PEP 8 for Python code style.
-   Prefer explicit, readable code over clever or overly compact
    constructs.
-   Use type annotations consistently.
-   Write docstrings using NumPy-style formatting.

------------------------------------------------------------------------

## Naming conventions

This project follows consistent naming conventions to clearly
distinguish **modules**, **types**, and **data-valued variables**. The
goal is to keep scientific code concise, readable, and free of name
collisions.

### Modules

Submodules are named after molecular data *domains* using short,
singular nouns:
```
automol.geom
automol.graph
automol.smiles
```
Modules act as **namespaces for algorithms and utilities**, not as
variable names.

**Rule:** Module names must not be used for data-valued variables.

------------------------------------------------------------------------

### Types (data models)

Data structures are defined as singular, capitalized class names:

``` python
Geometry
Graph
Smiles
```

These classes represent molecular data objects and define their schema
and validation.

------------------------------------------------------------------------

### Variables (instances of data types)

Variables holding instances of molecular data types use **short,
unambiguous abbreviations**, rather than full words or module names.

  Type              |Variable name
  ------------------|---------------
  `Geometry`        |`geo`
  `Graph`           |`gra`
  `Smiles`          |`smi`

Example:

``` python
from automol import geom, Geometry

geo = Geometry(["O", "H", "H"], coordinates)
com = geom.center_of_mass(geo)
```

**Rule:**
- `geom` refers to the **module** - `geo` refers to a **geometry
instance**

This distinction is used consistently throughout the codebase.

------------------------------------------------------------------------

### Functions vs methods

Algorithms operating on molecular data are implemented as **module-level
functions**, not instance methods:

``` python
def center_of_mass(geo: Geometry) -> FloatArray:
    ...
```

This keeps data models lightweight and separates data representation
from algorithms, following standard scientific Python practice.

------------------------------------------------------------------------

## Questions

If you have questions about contributing or design decisions, feel free
to open an issue for discussion.
