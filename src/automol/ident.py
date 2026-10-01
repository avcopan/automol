"""Molecular identities."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator
from rdkit import Chem

from . import geom, rd
from .utils.exc import AlgorithmAlreadyRegisteredError

if TYPE_CHECKING:
    from .geom import Geometry

# `Mapping` (rather than `dict`) is covariant in its value type, so a mapping
# of `Geometry` subclasses (e.g. `dict[str, SubGeometry]`) is accepted too.
OTHER_GEOS = Mapping[str, "Geometry"] | None


@runtime_checkable
class IdentityProtocol(Protocol):
    """Protocol for identity functions."""

    def __call__(self, geo: Geometry, other_geos: OTHER_GEOS = None) -> str:
        """Generate an identity string.

        Args:
            geo: Geometry to serialize.
            other_geos: Optional mapping of peer identities to geometries.

        Returns:
            str: Identity string.

        Raises:
            NotImplementedError: Always, for the protocol stub.
        """
        msg = "Identity function not implemented."
        raise NotImplementedError(msg)


@runtime_checkable
class GeometryProtocol(Protocol):
    """Protocol for geometry reconstruction functions."""

    def __call__(self, value: str) -> Geometry:
        """Reconstruct a geometry from an identity string.

        Args:
            value: Identity string.

        Returns:
            Geometry: Reconstructed geometry.

        Raises:
            NotImplementedError: Always, for the protocol stub.
        """
        msg = "Geometry function not implemented."
        raise NotImplementedError(msg)


class IdentityKind(StrEnum):
    """Category of molecular identity."""

    FORMULA = "formula"
    STEREOISOMER = "stereoisomer"
    CONFORMER = "conformer"
    ISOMER = "isomer"


def default_geometry_fn(value: str) -> Geometry:
    """Default geometry function that raises an error.

    Args:
        value: Identity string.

    Returns:
        Geometry: This function never returns successfully.

    Raises:
        NotImplementedError: Always, because no default geometry inversion is
            implemented.
    """
    msg = "Geometry function not implemented."
    raise NotImplementedError(msg)


def _check_signature(fn: Callable, *args: object, **kwargs: object) -> None:
    """Check that `fn` can be called with the given arguments.

    Args:
        fn: Callable to validate.
        *args: Positional arguments to bind.
        **kwargs: Keyword arguments to bind.

    Returns:
        None: Validation succeeds silently.

    Raises:
        ValueError: If `fn` has an incompatible signature for the supplied
            arguments.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # Signature not introspectable (e.g., builtins)
        return
    try:
        sig.bind(*args, **kwargs)
    except TypeError as err:
        msg = f"{fn!r} has an incompatible signature {sig}: {err}"
        raise ValueError(msg) from err


class Algorithm(BaseModel):
    """Boilerplate for Algorithm instances.

    Attributes:
        name (str): Unique algorithm name.
        kind (IdentityKind): Category of identity produced.
        parent_algorithm (Algorithm | None): Algorithm whose identity
            disambiguates this one's `other_geos`, if any.
        identity_fn (IdentityProtocol): Function
            ``(geo, other_geos=None) -> str`` generating the identity.
        geometry_fn (GeometryProtocol): Function ``(value) -> Geometry``
            inverting `identity_fn`, if supported.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    name: str
    kind: IdentityKind
    parent_algorithm: Algorithm | None = None

    identity_fn: IdentityProtocol
    geometry_fn: GeometryProtocol = default_geometry_fn

    @field_validator("identity_fn")
    @classmethod
    def validate_identity_fn(cls, fn: IdentityProtocol) -> IdentityProtocol:
        """Validate the signature of an identity function.

        Args:
            fn: Identity function to validate.

        Returns:
            IdentityProtocol: The validated identity function.
        """
        _check_signature(fn, None, None)
        _check_signature(fn, None, other_geos=None)
        return fn

    @field_validator("geometry_fn")
    @classmethod
    def validate_geometry_fn(cls, fn: GeometryProtocol) -> GeometryProtocol:
        """Validate the signature of a geometry reconstruction function.

        Args:
            fn: Geometry reconstruction function to validate.

        Returns:
            GeometryProtocol: The validated geometry function.
        """
        _check_signature(fn, "")
        return fn


class AlgorithmRegistry:
    """Central registry of all known identity algorithms."""

    algorithms: ClassVar[list[Algorithm]] = []

    @classmethod
    def register(
        cls,
        name: str,
        kind: IdentityKind,
        identity_fn: IdentityProtocol,
        geometry_fn: GeometryProtocol = default_geometry_fn,
        parent_algorithm: Algorithm | None = None,
    ) -> Algorithm:
        """Register an algorithm instance.

        Args:
            name: Unique algorithm name.
            kind: Category of identity produced.
            identity_fn: Function that generates the identity string.
            geometry_fn: Function that reconstructs a geometry from an identity.
            parent_algorithm: Algorithm whose identity disambiguates `other_geos`,
                if any.

        Returns:
            Algorithm: Registered algorithm instance.

        Raises:
            AlgorithmAlreadyRegisteredError: If an algorithm with `name` is
                already registered.
        """
        if any(name == a.name for a in cls.algorithms):
            msg = f"Algorithm {name!r} is already registered."
            raise AlgorithmAlreadyRegisteredError(msg)
        algorithm = Algorithm.model_validate(
            {
                "name": name,
                "kind": kind,
                "parent_algorithm": parent_algorithm,
                "identity_fn": identity_fn,
                "geometry_fn": geometry_fn,
            }
        )
        cls.algorithms.append(algorithm)
        return algorithm


def rdkit_inchi_geometry_fn(value: str) -> Geometry:
    """Generate a geometry from an InChI string with RDKit.

    Args:
        value: InChI string.

    Returns:
        Geometry: Geometry generated from `value`.
    """
    return geom.from_rdkit_mol(rd.mol.from_inchi(value, with_coords=True))


def rdkit_inchi_identity_fn(
    geo: Geometry,
    other_geos: OTHER_GEOS = None,  # noqa: ARG001
) -> str:
    """Generate an InChI string from a geometry with RDKit.

    Args:
        geo: Geometry to serialize.
        other_geos: Unused mapping of other geometries.

    Returns:
        str: InChI representation of `geo`.
    """
    return rd.mol.inchi(geom.rdkit_mol(geo))


rdkit_inchi = AlgorithmRegistry.register(
    name="rdkit inchi",
    kind=IdentityKind.STEREOISOMER,
    identity_fn=rdkit_inchi_identity_fn,
    geometry_fn=rdkit_inchi_geometry_fn,
)


def rdkit_smiles_geometry_fn(value: str) -> Geometry:
    """Generate a geometry from a SMILES string with RDKit.

    Args:
        value: SMILES string.

    Returns:
        Geometry: Geometry generated from `value`.
    """
    return geom.from_rdkit_mol(rd.mol.from_smiles(value, with_coords=True))


def rdkit_smiles_identity_fn(geo: Geometry, other_geos: OTHER_GEOS = None) -> str:
    """Generate SMILES from Geometry with RDKit.

    Args:
        geo: Geometry to serialize.
        other_geos: Mapping of SMILES strings to geometries used to reuse an
            existing key for an equivalent structure when possible.

    Returns:
        str: SMILES representation of `geo`.

    If a geometry in `other_geos` has the same InChI, its SMILES key is returned so
    that equivalent species share one (possibly non-canonical) SMILES. Otherwise,
    RDKit's canonical SMILES is returned.
    """
    mol = geom.rdkit_mol(geo)
    if other_geos:
        inchi = rd.mol.inchi(mol)
        for smi, other_geo in other_geos.items():
            if rdkit_inchi.identity_fn(other_geo) == inchi:
                return smi
    return Chem.MolToSmiles(Chem.RemoveAllHs(mol))


rdkit_smiles = AlgorithmRegistry.register(
    name="rdkit smiles",
    kind=IdentityKind.STEREOISOMER,
    parent_algorithm=rdkit_inchi,
    identity_fn=rdkit_smiles_identity_fn,
    geometry_fn=rdkit_smiles_geometry_fn,
)


def hill_formula_identity_fn(
    geo: Geometry,
    other_geos: OTHER_GEOS = None,  # noqa: ARG001
) -> str:
    """Render the molecular formula in Hill order.

    Args:
        geo: Geometry to serialize.
        other_geos: Unused mapping of other geometries.

    Returns:
        str: Molecular formula rendered in Hill order.

    With carbon present, C comes first, then H, then the remaining elements
    alphabetically. Without carbon, all elements (including H) are alphabetical.
    """
    counts = Counter(geo.symbols)

    ordered = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
        if "H" in counts:
            ordered.append(("H", counts.pop("H")))
    ordered.extend(sorted(counts.items()))

    return "".join(s if n == 1 else f"{s}{n}" for s, n in ordered)


hill_formula = AlgorithmRegistry.register(
    name="hill formula",
    kind=IdentityKind.FORMULA,
    parent_algorithm=rdkit_inchi,
    identity_fn=hill_formula_identity_fn,
)
