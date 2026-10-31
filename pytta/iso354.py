"""
PyTTa ISO 354 sound absorption in a reverberation room:
------------------------------------------------------
Equivalent sound absorption area and sound absorption coefficient of
ISO 354:2003 (Equations 5 to 9), with the power attenuation of air from
ISO 9613-1, and the Sabine and Eyring reverberation formulas.

    >>> alpha = pytta.iso354.absorption_coefficient(T1, T2, V, S, t1, t2)

"""
import numpy as np

from pytta import iso9613


def speed_of_sound(temperature):
    """Speed of sound in m/s, Equation (6): 331 + 0.6 t, for t from 15 to
    30 degrees Celsius."""
    return 331 + 0.6 * temperature


def power_attenuation(f, temperature=20., humidity=50., pressure=101.325):
    """Power attenuation coefficient m of air, in 1/m, from the attenuation
    alpha of ISO 9613-1 (dB/m): m = alpha / (10 lg e)."""
    return iso9613.attenuation(f, temperature, humidity, pressure) \
        / (10 * np.log10(np.e))


def absorption_area(T, V, c, m=0.):
    """Equivalent sound absorption area in m2 of a room of volume V (m3)
    with reverberation time T (s), Equations (5) and (7):
    A = 55.3 V / (c T) - 4 V m."""
    return 55.3 * V / (c * np.asarray(T)) - 4 * V * m


def absorption_coefficient(T1, T2, V, S, t1=20., t2=None, m1=0., m2=None):
    """
    Sound absorption coefficient of a test specimen of area S (m2),
    Equations (8) and (9), from the reverberation times of the empty room
    (T1) and with the specimen (T2), with the air temperatures t1 and t2
    (degrees Celsius) and power attenuations m1 and m2 of each measurement.
    t2 and m2 default to t1 and m1.
    """
    t2 = t1 if t2 is None else t2
    m2 = m1 if m2 is None else m2
    A1 = absorption_area(T1, V, speed_of_sound(t1), m1)
    A2 = absorption_area(T2, V, speed_of_sound(t2), m2)
    return (A2 - A1) / S


def sabine(V, A, c, m=0.):
    """Reverberation time in s by Sabine, the inverse of Equation (5):
    T = 55.3 V / (c (A + 4 V m))."""
    return 55.3 * V / (c * (np.asarray(A) + 4 * V * m))


def eyring(V, S, alpha, c, m=0.):
    """Reverberation time in s by Eyring, for room surface S (m2) with mean
    absorption coefficient alpha: T = 55.3 V / (c (-S ln(1 - alpha) +
    4 V m))."""
    return 55.3 * V / (c * (-S * np.log(1 - np.asarray(alpha)) + 4 * V * m))
