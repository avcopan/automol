"""Molecular identities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel, model_validator
from rdkit import Chem

from . import geom

if TYPE_CHECKING:
    from collections.abc import Callable

    from .geom import Geometry


@dataclass
class AlgorithmDef:
    """
    Descriptor for a single identity-generating algorithm.

    Attributes
    ----------
    kind
        Category of identity (e.g., "stereoisomer", "conformer").
    name
        Name of the registered algorithm that produced this identity.
    identity_fn
        Callable function to generate string identifier from geometry.
    geometry_fn
        Callable function to generate geometry from string identifier.
        `None` if the algorithm has no defined inverse.
    """

    kind: str
    name: str

    identity_fn: Callable[[Any], str]
    geometry_fn: Callable[[Any], Geometry] | None = None


class AlgorithmFns(ABC):
    """Boilerplate for Algorithm functions class."""

    @staticmethod
    @abstractmethod
    def identity_fn(geo: Geometry) -> str:
        """Generate an identifier string from a Geometry."""

    @staticmethod
    def geometry_fn(value: str) -> Geometry:
        """Instantiate a Geometry from an identifier string."""
        msg = f"Conversion of {value} to Geometry not implemented."
        raise NotImplementedError(msg)


class AlgorithmRegistry:
    """Central registry of all known identity algorithms."""

    _algorithms: ClassVar[dict[str, AlgorithmDef]] = {}

    @classmethod
    def register(
        cls, name: str, kind: str
    ) -> Callable[type[AlgorithmFns], type[AlgorithmFns]]:  # pyright: ignore[reportInvalidTypeForm]
        """Register identity_fn and geometry_fn as an AlgorithmDef."""

        def decorator(cls_: type[AlgorithmFns]) -> type[AlgorithmFns]:
            if name in cls._algorithms:
                msg = f"Algorithm {name!r} is already registered."
                raise ValueError(msg)
            cls._algorithms[name] = AlgorithmDef(
                name=name,
                kind=kind,
                identity_fn=staticmethod(cls_.identity_fn),
                geometry_fn=staticmethod(cls_.geometry_fn),
            )
            return cls_

        return decorator

    @classmethod
    def register_def(cls, alg: AlgorithmDef) -> None:
        """Directly register an AlgorithmDef instance."""
        if alg.name in cls._algorithms:
            msg = f"Algorithm {alg.name!r} is already registered."
            raise ValueError(msg)
        cls._algorithms[alg.name] = alg

    @classmethod
    def get(cls, name: str) -> AlgorithmDef:
        """Get an algorithm from registry."""
        try:
            return cls._algorithms[name]
        except KeyError:
            available = ", ".join(sorted(cls._algorithms))
            msg = f"Unknown algorithm {name!r}. Available: {available}"
            raise KeyError(msg) from None

    @classmethod
    def all_names(cls) -> list[str]:
        """Return all registered algorithms."""
        return sorted(cls._algorithms)

    @classmethod
    def names_for_kind(cls, kind: str) -> list[str]:
        """Return all registered algorithms for a kind."""
        return sorted(n for n, a in cls._algorithms.items() if a.kind == kind)


class Identity(BaseModel):
    """
    Molecular identity record.

    Parameters
    ----------
    kind
        Category of identity (e.g., "stereoisomer", "conformer").
    algorithm
        Name of the registered algorithm that produced this identity.
    value
        Resulting string identifier.
    """

    kind: str
    algorithm: str
    value: str

    @model_validator(mode="after")
    def _validate_algorithm_kind(self) -> Identity:
        alg = AlgorithmRegistry.get(self.algorithm)  # raises if unknown
        if alg.kind != self.kind:
            msg = f"Algorithm {self.algorithm!r} belongs to kind {alg.kind!r}."
            raise ValueError(msg)
        return self

    @classmethod
    def from_geometry(cls, geo: Geometry, *, algorithm: str) -> Self:
        """Return an Identity from a Geometry."""
        alg = AlgorithmRegistry.get(algorithm)
        value = alg.identity_fn(geo)
        return cls(kind=alg.kind, algorithm=algorithm, value=value)

    def geometry(self) -> Geometry:
        """Return a Geometry from Identity instance."""
        alg = AlgorithmRegistry.get(self.algorithm)
        if alg.geometry_fn:
            return alg.geometry_fn(self.value)
        raise NotImplementedError


@AlgorithmRegistry.register(name="rdkit inchi", kind="stereoisomer")
class RDKitInChI(AlgorithmFns):
    """Identify geometry with InChI using RDKit."""

    @staticmethod
    def identity_fn(geo: Geometry) -> str:
        """Generate InChI from Geometry with RDKit."""
        mol = geom.rdkit_mol(geo)
        mol_block = Chem.rdmolfiles.MolToMolBlock(mol)
        return Chem.inchi.MolBlockToInchi(mol_block)

    @staticmethod
    def geometry_fn(value: str) -> Geometry:
        """Generate Geometry from InChI with RDKit."""
        mol = Chem.MolFromInchi(value, sanitize=True, removeHs=False)
        mol = Chem.AddHs(mol)
        return geom.from_rdkit_mol(mol)


@AlgorithmRegistry.register(name="rdkit smiles", kind="stereoisomer")
class RDKitSMILES(AlgorithmFns):
    """Identify or generate geometry with SMILES using RDKit."""

    @staticmethod
    def identity_fn(geo: Geometry) -> str:
        """Generate SMILES from Geometry with RDKit."""
        mol = geom.rdkit_mol(geo)
        return Chem.MolToSmiles(Chem.RemoveAllHs(mol))

    @staticmethod
    def geometry_fn(value: str) -> Geometry:
        """Generate Geometry from SMILES with RDKit."""
        mol = Chem.MolFromSmiles(value)
        mol = Chem.AddHs(mol)
        return geom.from_rdkit_mol(mol)
