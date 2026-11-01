"""
PyTTa analytic:
---------------

Analytic sound fields, as the Monopole tools of the ITA-Toolbox
(applications/Analytic/Monopole): the free-field Green's function, a point
source over an impedance plane, a plane wave, and a plane wave scattered by
a sphere of given surface admittance.

Available functions:

    >>> pytta.analytic.green(f, source, points)
    >>> pytta.analytic.green_impedance_plane(f, source, points, Z)
    >>> pytta.analytic.plane_wave(f, direction, points)
    >>> p_scat, p_inc = pytta.analytic.plane_wave_on_sphere(f, radius, points)

Time dependence exp(j w t), so outgoing waves go as exp(-j k r). Points are
(n, 3) cartesian coordinates in m; results are (frequencies, points).

References
----------

Mechel, F. P. (2008). Formulas of Acoustics (2nd ed.). Springer.
Section on the scattering of a plane wave by a sphere.

Morse, P. M., & Ingard, K. U. (1968). Theoretical Acoustics. McGraw-Hill.
"""

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

__all__ = ['green', 'green_impedance_plane', 'plane_wave', 'plane_wave_on_sphere']

_C, _RHO = 343.2, 1.204


def _grid(f, points):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    P = np.atleast_2d(np.asarray(points, dtype=float))
    return f, P, 2 * np.pi * f[:, None]


def green(f, source, points, c=_C):
    """
    Free-field Green's function exp(-j k r) / (4 pi r) from source to each
    point, as ita_analytic_green_function without a plane.
    """
    f, P, w = _grid(f, points)
    r = np.linalg.norm(P - np.asarray(source, dtype=float), axis=1)[None, :]
    return np.exp(-1j * w / c * r) / (4 * np.pi * r)


def green_impedance_plane(f, source, points, Z, c=_C, rho=_RHO):
    """
    Point source over the plane z = 0 of surface impedance Z (Pa s/m, a
    number or one per frequency), as ita_analytic_green_function with an
    impedance plane: the free field plus the image source at (x0, y0, -z0)
    weighted by the plane-wave reflection factor at the specular angle,
    R = (Z cos(theta) - rho c) / (Z cos(theta) + rho c).

    The image with the plane-wave factor is the usual approximation; it is
    exact for a rigid (Z = inf) or a pressure-release (Z = 0) plane, and
    leaves out the ground wave of the exact solution.
    """
    f, P, w = _grid(f, points)
    s = np.asarray(source, dtype=float)
    image = s * np.array([1, 1, -1])
    r2 = np.linalg.norm(P - image, axis=1)[None, :]
    cos_t = (P[:, 2] + s[2])[None, :] / r2
    Z = np.broadcast_to(np.asarray(Z, dtype=complex).reshape(-1, 1) if np.ndim(Z) else Z,
                        (f.size, 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.where(np.isinf(Z), 1, (Z * cos_t - rho * c) / (Z * cos_t + rho * c))
    return green(f, s, P, c) + R * np.exp(-1j * w / c * r2) / (4 * np.pi * r2)


def plane_wave(f, direction, points, origin=(0, 0, 0), c=_C):
    """Plane wave of unit amplitude, exp(-j k u . (r - origin)), u the direction."""
    f, P, w = _grid(f, points)
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    return np.exp(-1j * w / c * ((P - np.asarray(origin, dtype=float)) @ u)[None, :])


def plane_wave_on_sphere(f, radius, points, admittance=0., c=_C, max_order=80):
    """
    Plane wave exp(-j k z) scattered by a sphere of radius a at the origin,
    as ita_analytic_plane_wave_on_sphere (series of Mechel, Formulas of
    Acoustics): with the normalized surface admittance beta (rho c times the
    admittance, 0 for a rigid sphere), the boundary condition
    dp/dr = j k beta p at r = a gives

        p_scat = sum_m (2m + 1) (-j)^m D_m P_m(cos theta) h_m(k r),
        D_m = -(j_m'(ka) - j beta j_m(ka)) / (h_m'(ka) - j beta h_m(ka)),

    with h_m = j_m - j y_m (outgoing for exp(j w t)).

    Parameters
    ----------
    f : array_like
        Frequencies (Hz).
    radius : float
    points : array_like
        (n, 3) points on or outside the sphere.
    admittance : complex or array_like
        beta, a number or one per frequency. Default 0.
    max_order : int
        Highest order of the series. Default 80.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Scattered and incident pressure, (frequencies, points).
    """
    f, P, w = _grid(f, points)
    r = np.linalg.norm(P, axis=1)
    if np.any(r < radius * (1 - 1e-12)):
        raise ValueError('points must lie on or outside the sphere.')
    cos_t = np.where(r > 0, P[:, 2] / np.where(r > 0, r, 1), 1)
    k = w[:, 0] / c
    beta = np.broadcast_to(np.asarray(admittance, dtype=complex), k.shape)[:, None]
    m = np.arange(max_order + 1)[None, :]
    ka = (k * radius)[:, None]
    j, jd = spherical_jn(m, ka), spherical_jn(m, ka, derivative=True)
    y, yd = spherical_yn(m, ka), spherical_yn(m, ka, derivative=True)
    h, hd = j - 1j * y, jd - 1j * yd
    D = -(jd - 1j * beta * j) / (hd - 1j * beta * h)          # (nf, orders)
    coef = (2 * m + 1) * (-1j) ** m                            # (1, orders)
    leg = np.array([eval_legendre(n, cos_t) for n in range(max_order + 1)])   # (orders, np)
    kr = k[:, None, None] * r[None, None, :]                   # (nf, 1, np)
    mm = np.arange(max_order + 1)[None, :, None]
    hr = spherical_jn(mm, kr) - 1j * spherical_yn(mm, kr)      # (nf, orders, np)
    p_scat = np.einsum('fm,mp,fmp->fp', D * coef, leg, hr)
    return p_scat, plane_wave(f, (0, 0, 1), P, c=c)
