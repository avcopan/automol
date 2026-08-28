"""Molecular identities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import BaseModel, model_validator
from rdkit import Chem

from . import geom
from .utils.exc import AlgorithmAlreadyRegisteredError, UnknownAlgorithmError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .geom import Geometry


class Algorithm(StrEnum):
    """
    Registered identity-generating algorithm, tagged with its kind.

    Attributes
    ----------
    kind
        Category of identity this algorithm produces (e.g., "stereoisomer").
    """

    kind: str

    def __new__(cls, value: str, kind: str) -> Self:
        """Construct an Algorithm member, attaching its kind."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.kind = kind
        return obj

    RDKIT_INCHI = ("rdkit inchi", "stereoisomer")
    RDKIT_SMILES = ("rdkit smiles", "stereoisomer")
    HILL_FORMULA = ("hill formula", "formula")


@dataclass
class AlgorithmDef:
    """
    Descriptor for a single identity-generating algorithm.

    Attributes
    ----------
    algorithm
        Registered algorithm that produced this identity.
    identity_fn
        Callable function to generate string identifier from geometry.
    geometry_fn
        Callable function to generate geometry from string identifier.
        `None` if the algorithm has no defined inverse.
    """

    algorithm: Algorithm

    identity_fn: Callable[[Geometry], str]
    geometry_fn: Callable[[str], Geometry] | None = None


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

    _algorithms: ClassVar[dict[Algorithm, AlgorithmDef]] = {}

    @classmethod
    def register(
        cls, algorithm: Algorithm
    ) -> Callable[[type[AlgorithmFns]], type[AlgorithmFns]]:
        """Register identity_fn and geometry_fn as an AlgorithmDef."""

        def decorator(cls_: type[AlgorithmFns]) -> type[AlgorithmFns]:
            if algorithm in cls._algorithms:
                msg = f"Algorithm {algorithm!r} is already registered."
                raise AlgorithmAlreadyRegisteredError(msg)
            cls._algorithms[algorithm] = AlgorithmDef(
                algorithm=algorithm,
                identity_fn=staticmethod(cls_.identity_fn),
                geometry_fn=staticmethod(cls_.geometry_fn),
            )
            return cls_

        return decorator

    @classmethod
    def register_def(cls, alg: AlgorithmDef) -> None:
        """Directly register an AlgorithmDef instance."""
        if alg.algorithm in cls._algorithms:
            msg = f"Algorithm {alg.algorithm!r} is already registered."
            raise AlgorithmAlreadyRegisteredError(msg)
        cls._algorithms[alg.algorithm] = alg

    @classmethod
    def get(cls, algorithm: Algorithm) -> AlgorithmDef:
        """Get an algorithm from registry."""
        try:
            return cls._algorithms[algorithm]
        except KeyError:
            available = ", ".join(sorted(cls._algorithms))
            msg = f"Unknown algorithm {algorithm!r}. Available: {available}"
            raise UnknownAlgorithmError(msg) from None

    @classmethod
    def all_algorithms(cls) -> list[Algorithm]:
        """Return all registered algorithms."""
        return sorted(cls._algorithms)

    @classmethod
    def algorithms_for_kind(cls, kind: str) -> list[Algorithm]:
        """Return all registered algorithms for a kind."""
        return sorted(a for a in cls._algorithms if a.kind == kind)


class Identity(BaseModel):
    """
    Molecular identity record.

    Parameters
    ----------
    algorithm
        Registered algorithm that produced this identity.
    value
        Resulting string identifier.
    kind
        Category of identity (e.g., "stereoisomer", "conformer"). Must match
        `algorithm.kind`; prefer `from_geometry` or `from_value` over setting
        this directly.
    """

    algorithm: Algorithm
    value: str
    kind: str

    @model_validator(mode="after")
    def _validate_algorithm_kind(self) -> Identity:
        if self.kind != self.algorithm.kind:
            msg = (
                f"Algorithm {self.algorithm!r} belongs to kind "
                f"{self.algorithm.kind!r}, not {self.kind!r}."
            )
            # Pydantic only wraps ValueError/TypeError/AssertionError from
            # model validators into a ValidationError; anything else bypasses
            # that pipeline entirely, so this must stay a plain ValueError.
            raise ValueError(msg)
        return self

    @classmethod
    def from_geometry(cls, geo: Geometry, *, algorithm: Algorithm) -> Self:
        """Return an Identity from a Geometry, by algorithm alone."""
        alg = AlgorithmRegistry.get(algorithm)
        value = alg.identity_fn(geo)
        return cls.from_value(value, algorithm=algorithm)

    @classmethod
    def from_value(cls, value: str, *, algorithm: Algorithm) -> Self:
        """Return an Identity from an already-computed value, by algorithm alone."""
        return cls(algorithm=algorithm, value=value, kind=algorithm.kind)

    def geometry(self) -> Geometry:
        """Return a Geometry from Identity instance."""
        alg = AlgorithmRegistry.get(self.algorithm)
        if alg.geometry_fn:
            return alg.geometry_fn(self.value)
        raise NotImplementedError


@AlgorithmRegistry.register(Algorithm.RDKIT_INCHI)
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


@AlgorithmRegistry.register(Algorithm.RDKIT_SMILES)
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


@AlgorithmRegistry.register(Algorithm.HILL_FORMULA)
class HillFormula(AlgorithmFns):
    """Identify geometry with its molecular formula in Hill order."""

    @staticmethod
    def identity_fn(geo: Geometry) -> str:
        """Render the molecular formula in Hill order."""
        counts = Counter(s.capitalize() for s in geo.symbols)

        ordered = []
        if "C" in counts:
            ordered.append(("C", counts.pop("C")))
        if "H" in counts:
            ordered.append(("H", counts.pop("H")))
        ordered.extend(sorted(counts.items(), key=lambda x: x[0]))

        return "".join(s if n == 1 else f"{s}{n}" for s, n in ordered)
