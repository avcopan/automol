"""Utility tests."""

import pytest

from automol.utils.exc import (
    GeometryConversionError,
    HashGenerationError,
    XYZFormatError,
)


def test__geometry_conversion_error() -> None:
    """Test GeometryConversionError message."""
    msg = "bad conversion"
    with pytest.raises(GeometryConversionError, match=msg):
        raise GeometryConversionError(msg)


def test__hash_generation_error() -> None:
    """Test HashGenerationError message and hashable instance."""
    msg = "bad hash"
    with pytest.raises(HashGenerationError) as exc_info:
        raise HashGenerationError(msg, 42)

    assert exc_info.value.args == (msg, 42)


def test__xyz_format_error() -> None:
    """Test XYZFormatError message."""
    msg = "bad xyz"
    with pytest.raises(XYZFormatError, match=msg):
        raise XYZFormatError(msg)
