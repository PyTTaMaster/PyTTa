"""
PyTTa ISO 9613-1 air absorption:
-------------------------------
Attenuation of sound by atmospheric absorption of ISO 9613-1:1993, for pure
tones, from temperature, relative humidity and pressure.

    >>> pytta.iso9613.attenuation(1000, 20, 50)     # dB/m

The equations follow ISO 9613-1, as reproduced in Bies and Hansen,
Engineering Noise Control (5th ed., Equations 5.69 to 5.72 and 5.75).
"""
import numpy as np

_T0 = 293.15        # K, reference temperature
_T01 = 273.16       # K, triple point of water
_PR = 101.325       # kPa, reference pressure


def attenuation(f, temperature=20., humidity=50., pressure=101.325):
    """
    Attenuation coefficient in dB/m at frequency f (Hz) for air at
    `temperature` (degrees Celsius), relative `humidity` (%) and ambient
    `pressure` (kPa). Valid, per the standard, for molar concentrations of
    water vapour of 0.05 % to 5 % and temperatures of -20 to 50 degrees
    Celsius.
    """
    f = np.asarray(f, dtype=float)
    T = temperature + 273.15
    pa = pressure / _PR
    psat = 10 ** (-6.8346 * (_T01 / T) ** 1.261 + 4.6151)
    h = humidity * psat / pa                  # molar concentration, %
    frO = pa * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pa * (T / _T0) ** -0.5 * (
        9 + 280 * h * np.exp(-4.170 * ((T / _T0) ** (-1 / 3) - 1)))
    return 8.686 * f ** 2 * (
        1.84e-11 / pa * (T / _T0) ** 0.5
        + (T / _T0) ** -2.5 * (0.01275 * np.exp(-2239.1 / T)
                               / (frO + f ** 2 / frO)
                               + 0.1068 * np.exp(-3352.0 / T)
                               / (frN + f ** 2 / frN)))
