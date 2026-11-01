# -*- coding: utf-8 -*-
"""
Room acoustic parameters of open-plan offices, ISO 3382-3:2012 (read in its
Brazilian adoption, ABNT NBR ISO 3382-3): the A-weighted speech level at
each position from measurements with a calibrated omnidirectional source
(Eq. 1 to 4), the spatial decay rate of speech D2,S and the level at 4 m
Lp,A,S,4m (Eq. 5), and the distraction and privacy distances rD and rP from
the speech transmission index along the measurement line.

The STI at each position is computed separately, by the full method of IEC
60268-16, e.g. with pytta.iec60268.sti.

>>> LpAS = pytta.iso3382_3.speech_levels(Lw_source, Lp_source)
>>> D2S, Lp4m = pytta.iso3382_3.spatial_decay(r, LpAS)
>>> rD = pytta.iso3382_3.distance_at_sti(r, sti, 0.5)

"""

import numpy as np

__all__ = ['OCTAVES', 'SPEECH_LW', 'SPEECH_LP_1M', 'A_WEIGHTING', 'source_level_1m',
           'speech_levels', 'spatial_decay', 'distance_at_sti']

# Table 1: normal unisex speech in the octave bands 125 Hz to 8 kHz.
OCTAVES = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
SPEECH_LW = np.array([60.9, 65.3, 69.0, 63.0, 55.8, 49.8, 44.5])        # dB re 1 pW
SPEECH_LP_1M = np.array([49.9, 54.3, 58.0, 52.0, 44.8, 38.8, 33.5])     # omnidirectional, dB
A_WEIGHTING = np.array([-16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])


def source_level_1m(Lw):
    """Free field level at 1 m of a source of octave band power levels Lw,
    Lw + 10 lg(1 / (4 pi 1^2)), about Lw - 11 dB (Eq. 1)."""
    return np.asarray(Lw, dtype=float) + 10 * np.log10(1 / (4 * np.pi))


def speech_levels(Lw_source, Lp_source, speech=SPEECH_LP_1M):
    """
    A-weighted level of normal speech at each measurement position,
    Lp,A,S,n (Eq. 2 to 4). The attenuation of each octave band from 1 m in
    free field to position n, Dn,i = Lp,Ls,1m,i - Lp,Ls,n,i, measured with
    the source of power levels Lw_source (7 octave bands), is applied to the
    speech levels at 1 m of Table 1, and the bands are A-weighted and summed.
    Lp_source holds the levels of the source at the positions, shape (N, 7).
    """
    D = source_level_1m(Lw_source) - np.atleast_2d(np.asarray(Lp_source, dtype=float))
    LpS = np.asarray(speech, dtype=float) - D
    return 10 * np.log10(np.sum(10 ** ((LpS + A_WEIGHTING) / 10), axis=1))


def spatial_decay(r, LpAS, rmin=2., rmax=16.):
    """
    Spatial decay rate of speech per distance doubling D2,S and the speech
    level at 4 m Lp,A,S,4m from the regression line of Lp,A,S,n on lg(r / 1 m)
    over the positions between rmin and rmax (Eq. 5, 3.3 and 6.2).
    """
    r, L = np.asarray(r, dtype=float), np.asarray(LpAS, dtype=float)
    keep = (r >= rmin) & (r <= rmax)
    slope, intercept = np.polyfit(np.log10(r[keep]), L[keep], 1)
    return -np.log10(2) * slope, intercept + slope * np.log10(4.)


def distance_at_sti(r, sti, limit=0.5):
    """
    Distance at which the regression line of the STI on the distance (linear
    axis) falls to limit: the distraction distance rD for 0.50 and the
    privacy distance rP for 0.20 (3.6, 3.7 and 6.3). nan when the line does
    not fall.
    """
    slope, intercept = np.polyfit(np.asarray(r, dtype=float), np.asarray(sti, dtype=float), 1)
    return (limit - intercept) / slope if slope < -1e-12 else np.nan
