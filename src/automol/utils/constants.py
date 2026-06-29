"""Constant values."""

import numpy as np
import pint

ureg = pint.UnitRegistry()

# Constants
## Angular
RADIAN: pint.Quantity = 1 * ureg.radian
DEGREE: pint.Quantity = 1 * ureg.degree

## Mass
DALTON: pint.Quantity = 1 * ureg.amu
KILOGRAM: pint.Quantity = 1 * ureg.kilogram

## Distance
BOHR: pint.Quantity = 1 * ureg.bohr
ANGSTROM: pint.Quantity = 1 * ureg.angstrom
METER: pint.Quantity = 1 * ureg.meter
CENTIMETER: pint.Quantity = 1 * ureg.centimeter

## Time
SECOND: pint.Quantity = 1 * ureg.second

## Velocity
METERS_PER_SECOND: pint.Quantity = METER / SECOND
CENTIMETERS_PER_SECOND: pint.Quantity = CENTIMETER / SECOND

## Energy
HARTREE: pint.Quantity = 1 * ureg.hartree
JOULE: pint.Quantity = 1 * ureg.joule

# Physical
C: pint.Quantity = 1 * ureg.speed_of_light
WAVENUMBER: pint.Quantity = 1 / CENTIMETER

# Conversions
## Angular
RADIANS_TO_DEGREES: float = RADIAN.m_as(DEGREE)
DEGREES_TO_RADIANS: float = DEGREE.m_as(RADIAN)

## Mass
DALTON_TO_KILOGRAMS: float = DALTON.m_as(KILOGRAM)
KILOGRAMS_TO_DALTON: float = KILOGRAM.m_as(DALTON)

## Distance
BOHR_TO_METERS: float = BOHR.m_as(METER)
METERS_TO_BOHR: float = METER.m_as(BOHR)
BOHR_TO_ANGSTROM: float = BOHR.m_as(ANGSTROM)
ANGSTROM_TO_BOHR: float = ANGSTROM.m_as(BOHR)

## Energy
HARTREE_TO_JOULES: float = HARTREE.m_as(JOULE)
JOULES_TO_HARTREE: float = JOULE.m_as(HARTREE)
with ureg.context("spectroscopy"):
    WAVENUMBER_TO_HARTREE: float = WAVENUMBER.m_as(HARTREE)

## Physical
C_METERS_PER_SECOND: float = C.m_as(METERS_PER_SECOND)
C_CENTIMETERS_PER_SECOND: float = C.m_as(CENTIMETERS_PER_SECOND)

## Frequency
VIBRATIONAL_FORCE_TO_INV_CM_FREQUENCY: float = np.sqrt(
    HARTREE_TO_JOULES / (DALTON_TO_KILOGRAMS * BOHR_TO_METERS * BOHR_TO_METERS)
) / (C_CENTIMETERS_PER_SECOND * 2 * np.pi)
