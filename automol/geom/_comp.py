""" some comparison functions
"""
import numpy
from ._core import symbols as _symbols
from ._core import coordinates as _coordinates
from ._repr import coulomb_spectrum as _coulomb_spectrum


def almost_equal(geo1, geo2, rtol=1e-5):
    """ are these geometries almost equal?
    """
    ret = False
    if _symbols(geo1) == _symbols(geo2):
        ret = numpy.allclose(_coordinates(geo1), _coordinates(geo2), rtol=rtol)
    return ret


def almost_equal_coulomb_spectrum(geo1, geo2, rtol=1e-5):
    """ do these geometries have similar coulomb spectrums?
    """
    ret = numpy.allclose(_coulomb_spectrum(geo1), _coulomb_spectrum(geo2),
                         rtol=rtol)
    return ret