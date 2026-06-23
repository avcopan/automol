"""Automol exceptions."""


class GeometryConversionError(Exception):
    """Raise an error when Geometry conversion is not successful."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class HashGenerationError(Exception):
    """Raise an error when object hashing is not successful."""

    def __init__[T](self, message: str, hashable_instance: T) -> None:
        super().__init__(message, hashable_instance)


class XYZFormatError(Exception):
    """Raise an error when xyz format fails to parse."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
