# -*- coding: utf-8 -*-
"""
Normal incidence sound transmission in a four-microphone tube by the
transfer matrix method of ASTM E2611-09, as ita_impedance_tube_transmission
and ita_kundt_calc_impedance_transmission of the ITA-Toolbox.

The transfer functions H have shape (4, nFreqs), each relative to the
same reference: microphones 1 and 2 upstream of the specimen, at l1 + s1
and l1 from its front face, and 3 and 4 downstream, at l2 and l2 + s2 from
the same front face, so that l2 includes the thickness d (Fig. 4). The transfer matrix T relates pressure and
particle velocity on the front face to those on the back face (Eq 22),
with shape (nFreqs, 2, 2).

>>> T = pytta.e2611.transfer_matrix(f, Ha, Hb, l1, s1, l2, s2, d)
>>> TL = pytta.e2611.transmission_loss(T, f, d)

"""

import numpy as np

__all__ = ['speed_of_sound', 'air_density', 'wave_components',
           'transfer_matrix', 'transmission_loss', 'reflection_coefficient',
           'absorption_coefficient', 'material_properties']


def speed_of_sound(temperature):
    """Speed of sound in m/s at temperature in degrees Celsius, Eq 4."""
    return 20.047 * np.sqrt(273.15 + temperature)


def air_density(temperature, pressure=101.325):
    """Air density in kg/m3 at temperature (degrees Celsius) and pressure
    (kPa), Eq 5."""
    return 1.290 * (pressure / 101.325) * (273.15 / (273.15 + temperature))


def wave_components(f, H, l1, s1, l2, s2, c=343.):
    """Amplitudes A, B of the waves upstream and C, D downstream of the
    specimen from the transfer functions H of the four microphones,
    Eq 17 to 20."""
    k = 2 * np.pi * np.asarray(f) / c
    H1, H2, H3, H4 = H
    A = 1j * (H1 * np.exp(-1j * k * l1) - H2 * np.exp(-1j * k * (l1 + s1))) / (2 * np.sin(k * s1))
    B = 1j * (H2 * np.exp(1j * k * (l1 + s1)) - H1 * np.exp(1j * k * l1)) / (2 * np.sin(k * s1))
    C = 1j * (H3 * np.exp(1j * k * (l2 + s2)) - H4 * np.exp(1j * k * l2)) / (2 * np.sin(k * s2))
    D = 1j * (H4 * np.exp(-1j * k * l2) - H3 * np.exp(-1j * k * (l2 + s2))) / (2 * np.sin(k * s2))
    return A, B, C, D


def _faces(f, H, l1, s1, l2, s2, d, c, rho):
    # Pressure and particle velocity on both faces, Eq 21.
    A, B, C, D = wave_components(f, H, l1, s1, l2, s2, c)
    k = 2 * np.pi * np.asarray(f) / c
    p0, u0 = A + B, (A - B) / (rho * c)
    pd = C * np.exp(-1j * k * d) + D * np.exp(1j * k * d)
    ud = (C * np.exp(-1j * k * d) - D * np.exp(1j * k * d)) / (rho * c)
    return p0, u0, pd, ud


def transfer_matrix(f, Ha, Hb=None, l1=0., s1=0., l2=0., s2=0., d=0., c=343., rho=1.21):
    """
    Transfer matrix of the specimen from the transfer functions Ha and Hb
    measured with two different terminations (two-load method, Eq 22), or
    from Ha alone for a symmetric specimen (one-load method, Eq 24).
    """
    p0a, u0a, pda, uda = _faces(f, Ha, l1, s1, l2, s2, d, c, rho)
    if Hb is None:
        den = p0a * uda + pda * u0a
        T = [(pda * uda + p0a * u0a) / den, (p0a ** 2 - pda ** 2) / den,
             (u0a ** 2 - uda ** 2) / den, (pda * uda + p0a * u0a) / den]
    else:
        p0b, u0b, pdb, udb = _faces(f, Hb, l1, s1, l2, s2, d, c, rho)
        den = pda * udb - pdb * uda
        T = [(p0a * udb - p0b * uda) / den, (p0b * pda - p0a * pdb) / den,
             (u0a * udb - u0b * uda) / den, (pda * u0b - pdb * u0a) / den]
    return np.stack(T, axis=-1).reshape(np.shape(f) + (2, 2))


def transmission_loss(T, f, d, c=343., rho=1.21):
    """Normal incidence transmission loss in dB of an anechoic-backed
    specimen, Eq 25 and 26. Returns (TL, complex transmission coefficient)."""
    k = 2 * np.pi * np.asarray(f) / c
    t = 2 * np.exp(1j * k * d) / (T[..., 0, 0] + T[..., 0, 1] / (rho * c)
                                  + rho * c * T[..., 1, 0] + T[..., 1, 1])
    return 20 * np.log10(1 / np.abs(t)), t


def reflection_coefficient(T, c=343., rho=1.21):
    """Reflection coefficient of the hard-backed specimen, Eq 27."""
    return (T[..., 0, 0] - rho * c * T[..., 1, 0]) / (T[..., 0, 0] + rho * c * T[..., 1, 0])


def absorption_coefficient(T, c=343., rho=1.21):
    """Absorption coefficient of the hard-backed specimen, Eq 28."""
    return 1 - np.abs(reflection_coefficient(T, c, rho)) ** 2


def material_properties(T, d):
    """Wavenumber k' = arccos(T11) / d (Eq 29) and characteristic impedance
    z = sqrt(T12 / T21) (Eq 30) in the material."""
    return np.arccos(T[..., 0, 0] + 0j) / d, np.sqrt(T[..., 0, 1] / T[..., 1, 0] + 0j)
