"""
PyTTa ISO 10534-2 impedance tube:
--------------------------------
Normal-incidence reflection factor, absorption coefficient and impedance
of a sample in an impedance tube, by the two-microphone transfer function
method of ISO 10534-2:1998.

    >>> f, H12 = pytta.iso10534.transfer_function(mic1, mic2)
    >>> r = pytta.iso10534.reflection_factor(H12, f, x1, s, c0)
    >>> alpha = pytta.iso10534.absorption(r)

Microphone 1 is the one further from the sample, at x1 from it; microphone
2 is s closer. The time convention is that of numpy.fft, exp(+j w t), the
same as the standard's.
"""
import numpy as np


def speed_of_sound(temperature):
    """Speed of sound in m/s at `temperature` in degrees Celsius,
    Equation (5): c0 = 343.2 sqrt(T / 293), T in kelvin."""
    return 343.2 * np.sqrt((temperature + 273.15) / 293)


def frequency_range(d, s, c0):
    """
    Working frequency range (f_low, f_up) in Hz of a circular tube of inner
    diameter d with microphones s apart, both in m: f_up from Equations (2)
    and (4) (f_up d < 0.58 c0, f_up s < 0.45 c0), and f_low where s is 5 %
    of the wavelength, the general guide of 4.2.
    """
    return 0.05 * c0 / s, min(0.58 * c0 / d, 0.45 * c0 / s)


def transfer_function(mic1, mic2):
    """Frequencies and transfer function H12 = P2 / P1 from the recordings
    or impulse responses of the two microphone positions (first channel of
    each SignalObj)."""
    p1 = np.asarray(mic1.timeSignal[:, 0], dtype=float)
    p2 = np.asarray(mic2.timeSignal[:, 0], dtype=float)
    return np.fft.rfftfreq(p1.size, 1 / mic1.samplingRate)[1:], \
        (np.fft.rfft(p2) / np.fft.rfft(p1))[1:]


def reflection_factor(H12, f, x1, s, c0, calibration=None):
    """
    Normal-incidence reflection factor, Equation (17):
    r = (H12 - HI) / (HR - H12) exp(2 j k x1), with HI = exp(-j k s) and
    HR = exp(j k s) (D.5, D.6). `calibration` is the factor Hc of 7.5.2,
    which divides H12, if the microphones were not interchanged.
    """
    H12 = np.asarray(H12) if calibration is None \
        else np.asarray(H12) / calibration
    k = 2 * np.pi * np.asarray(f) / c0
    return (H12 - np.exp(-1j * k * s)) / (np.exp(1j * k * s) - H12) \
        * np.exp(2j * k * x1)


def absorption(r):
    """Normal-incidence absorption coefficient, Equation (18):
    1 - |r|^2."""
    return 1 - np.abs(r) ** 2


def impedance_ratio(r):
    """Specific acoustic impedance ratio Z / (rho c0), Equation (19):
    (1 + r) / (1 - r)."""
    return (1 + np.asarray(r)) / (1 - np.asarray(r))
