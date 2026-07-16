"""Automol exceptions."""


class AlgorithmAlreadyRegisteredError(Exception):
    """Raise an error when an identity algorithm is already registered."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ElementNotFoundError(Exception):
    """Raise an error when an element cannot be found by atomic number or symbol."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class GeometryConversionError(Exception):
    """Raise an error when Geometry conversion is not successful."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class HashGenerationError(Exception):
    """Raise an error when object hashing is not successful."""

    def __init__[T](self, message: str, hashable_instance: T) -> None:
        super().__init__(message, hashable_instance)


class UnknownAlgorithmError(Exception):
    """Raise an error when an identity algorithm is not registered."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class XYZFormatError(Exception):
    """Raise an error when xyz format fails to parse."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
