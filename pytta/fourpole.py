# -*- coding: utf-8 -*-
"""
Four-pole (two-port) matrices of lumped mechanical and acoustic elements,
as itaFourpole and ita_make_fourpole of the ITA-Toolbox.

The transmission matrix A of an element relates the force (or pressure) and
the velocity (or volume velocity) at its input to those at its output,

    [F1, v1] = A @ [F2, v2],

with v2 leaving the element at the output. The matrices are arrays of shape
(nFreqs, 2, 2), and a chain of elements is the product of their matrices.

>>> f = np.linspace(1, 1000, 1000)
>>> A = pytta.fourpole.chain(pytta.fourpole.element('mass', 0.1, f),
...                          pytta.fourpole.element('spring', 1e-4, f))
>>> Zin = pytta.fourpole.input_impedance(A)   # output blocked

"""

import numpy as np

__all__ = ['element', 'chain', 'input_impedance', 'a_to_z', 'z_to_a']


def element(kind, value, freqs):
    """
    Transmission matrix of one element at the frequencies freqs (Hz):

        * 'mass': a mass of value kg, moving with both ports, A12 = j omega m;
        * 'spring': a spring of compliance value m/N, A21 = j omega n;
        * 'damper': a dashpot of resistance value N s/m, A21 = 1 / r;
        * 'series': an impedance value (scalar or one per frequency) in
          series, A12 = Z;
        * 'shunt': an impedance value to ground, A21 = 1 / Z;
        * 'transformer': A = diag(value, 1 / value);
        * 'gyrator': A = [[0, value], [1 / value, 0]].
    """
    jw = 2j * np.pi * np.asarray(freqs, dtype=float)
    value = np.broadcast_to(np.asarray(value, dtype=complex), jw.shape)
    one, zero = np.ones_like(value), np.zeros_like(value)
    a = {'mass': (one, jw * value, zero, one),
         'spring': (one, zero, jw * value, one),
         'damper': (one, zero, 1 / value, one),
         'series': (one, value, zero, one),
         'shunt': (one, zero, 1 / value, one),
         'transformer': (value, zero, zero, 1 / value),
         'gyrator': (zero, value, 1 / value, zero)}
    if kind not in a:
        raise ValueError(f'kind must be one of {", ".join(a)}.')
    return np.stack(a[kind], axis=-1).reshape(jw.shape + (2, 2))


def chain(*matrices):
    """Transmission matrix of elements connected one after the other."""
    result = matrices[0]
    for m in matrices[1:]:
        result = result @ m
    return result


def input_impedance(A, load=None):
    """
    F1 / v1 of the chain A with the impedance load at its output; None is a
    blocked output (v2 = 0) and 0 a free one (F2 = 0).
    """
    if load is None:
        return A[..., 0, 0] / A[..., 1, 0]
    return (A[..., 0, 0] * load + A[..., 0, 1]) / (A[..., 1, 0] * load + A[..., 1, 1])


def a_to_z(A):
    """
    Impedance matrix, [F1, F2] = Z @ [v1, -v2], of the transmission matrix
    A, that is with both velocities entering the element, as
    itaFourpole.A2Z; Z12 = Z21 for a reciprocal element (det A = 1).
    """
    a11, a12, a21, a22 = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    det = a11 * a22 - a12 * a21
    return np.stack([a11, det, np.ones_like(a11), a22], axis=-1).reshape(A.shape) / a21[..., None, None]


def z_to_a(Z):
    """Transmission matrix of the impedance matrix Z, the inverse of a_to_z."""
    z11, z12, z21, z22 = Z[..., 0, 0], Z[..., 0, 1], Z[..., 1, 0], Z[..., 1, 1]
    det = z11 * z22 - z12 * z21
    return np.stack([z11, det, np.ones_like(z11), z22], axis=-1).reshape(Z.shape) / z21[..., None, None]
