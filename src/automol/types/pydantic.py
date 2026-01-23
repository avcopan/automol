"""Pydantic types."""

from typing import Annotated

import numpy as np
from numpy import typing as npt
from pydantic import BeforeValidator, PlainSerializer
from pydantic.functional_validators import SkipValidation

from .core import Coordinates, FloatArray


# Float array field
def _float_array_validator(obj: object) -> FloatArray:
    return np.asarray(obj, dtype=np.float64)


def _float_array_serializer(arr: FloatArray) -> list:
    return arr.tolist()


FloatArrayField = Annotated[
    SkipValidation[npt.ArrayLike],
    BeforeValidator(_float_array_validator),
    PlainSerializer(_float_array_serializer, return_type=list),
]


# Coordinates field
def _coordinates_validator(obj: object) -> Coordinates:
    arr = _float_array_validator(obj)

    if arr.ndim != 2 or arr.shape[-1] != 3:  # noqa: PLR2004
        msg = f"Expected array of shape (N, 3) but got {arr.shape}."
        raise ValueError(msg)

    return arr


CoordinatesField = Annotated[
    SkipValidation[npt.ArrayLike],
    BeforeValidator(_coordinates_validator),
    PlainSerializer(_float_array_serializer, return_type=list),
]
