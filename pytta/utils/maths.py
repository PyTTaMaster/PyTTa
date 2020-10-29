#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue May  5 00:34:36 2020

@author: joaovitor
"""

import numpy as np
import numba as nb


@nb.njit
def maxabs(arr: np.array) -> int or float:
    """
    Maximum of the absolute of array values.

    Arguments
    ---------
        arr (np.array): Array of data.

    Returns
    -------
        int or float: Maximum of the absolute values.

    """
    return np.max(np.abs(arr))


@nb.njit
def arr2rms(arr: np.array) -> float:
    """
    Root of the mean of a squared array.

    Arguments
    ---------
        arr (np.array): Array of data.

    Returns
    -------
        float: RMS of data.

    """
    return (np.mean(arr**2))**0.5


@nb.njit
def rms2dB(rms: float, power: bool = False, ref: float = 1.0) -> float:
    """
    RMS to decibel.

    Arguments
    ---------
        rms (float): The value to be scaled.
        power (bool, optional): If array is a power signal. Defaults to False.
        ref (float, optional): Reference value for decibel scale. Defaults to 1.0.

    Returns
    -------
        float: Decibel scaled value.

    """
    return (10 if power else 20) * np.log10(rms / ref)


@nb.njit
def arr2dB(arr: np.array, power: bool = False, ref: float = 1.) -> float:
    """
    Calculate the decibel level of an array of data.

    Arguments
    ---------
        arr (np.array): Array of data.
        power (bool, optional): If it's power data. Defaults to False.
        ref (float, optional): Decibel reference. Defaults to 1..

    Returns
    -------
        float: Decibel scaled value.

    """
    return rms2dB(arr2rms(arr), power, ref)


def fft_degree(timeLength: float = 0, samplingRate: int = 1) -> float:
    """
    Power-of-two value that can be used to calculate the total number of samples of the signal.

        >>> numSamples = 2**fftDegree

    Parameters
    ----------
        * timeLength (float = 0):
            Value, in seconds, of the time duration of the signal or
            recording.

        * samplingRate (int = 1):
            Value, in samples per second, that the data will be captured
            or emitted.

    Returns
    -------
        fftDegree (float = 0):
            Power of 2 that can be used to calculate number of samples.

    """
    return np.log2(round(timeLength*samplingRate))
