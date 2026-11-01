"""
PyTTa absorbers:
----------------

Surface impedance and absorption of layered absorbers by the transfer
matrix method, as the Impedance Calculator of the ITA-Toolbox
(applications/Analytic/Impedance_Calculator): layers of air, porous
material (Delany-Bazley, Miki or Komatsu), microperforated panels (Maa) and
limp masses on a rigid backing, at normal or oblique incidence, and the
random-incidence absorption of a locally reacting surface.

Available functions:

    >>> Zc, k = pytta.absorbers.porous(f, sigma, model='miki')
    >>> Zs = pytta.absorbers.surface_impedance(f, layers)
    >>> alpha = pytta.absorbers.absorption(Zs)
    >>> alpha_d = pytta.absorbers.random_incidence_absorption(f, layers)

Time dependence exp(j w t). Impedances in Pa s/m, flow resistivity sigma in
Pa s/m^2, lengths in m.

References
----------

Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
absorbent materials. Applied Acoustics, 3(2), 105-116.

Miki, Y. (1990). Acoustical properties of porous materials. Modifications of
Delany-Bazley models. Journal of the Acoustical Society of Japan (E), 11(1),
19-24.

Komatsu, T. (2008). Improvement of the Delany-Bazley and Miki models for
fibrous sound-absorbing materials. Acoustical Science and Technology, 29(2),
121-129.

Maa, D.-Y. (1998). Potential of microperforated panel absorber. The Journal
of the Acoustical Society of America, 104(5), 2861-2866.

Cox, T. J., & D'Antonio, P. (2009). Acoustic Absorbers and Diffusers (2nd
ed.). Taylor & Francis.
"""

import numpy as np

__all__ = ['porous', 'mpp_impedance', 'surface_impedance', 'absorption',
           'random_incidence_absorption']

_RHO, _C, _ETA = 1.204, 343.2, 1.81e-5


def porous(f, sigma, model='miki', rho=_RHO, c=_C):
    """
    Characteristic impedance Zc (Pa s/m) and wavenumber k (1/m, with a
    negative imaginary part) of a porous material of flow resistivity sigma
    (Pa s/m^2), by the empirical models of Delany and Bazley (1970), Miki
    (1990) or Komatsu (2008):

    * 'delany_bazley': X = rho f / sigma, Zc = rho c (1 + 0.0571 X^-0.754
      - j 0.087 X^-0.732), k = w/c (1 + 0.0978 X^-0.700 - j 0.189 X^-0.595);
    * 'miki': X = 1000 f / sigma, Zc = rho c (1 + 5.50 X^-0.632 - j 8.43
      X^-0.632), k = w/c (1 + 7.81 X^-0.618 - j 11.41 X^-0.618), the
      constants of Miki's paper, Eqs. (27) and (28) (often rounded in SI
      units to 0.0699, 0.107, 0.109 and 0.160 with f / sigma);
    * 'komatsu': u = 2 - lg(f / sigma), Zc = rho c (1 + 0.00027 u^6.2 -
      j 0.0047 u^4.1), k = w/c (1 + 0.0004 u^6.2 - j 0.0069 u^4.1), for
      f / sigma up to 100 (u >= 0), where it reaches air; above, nan.
    """
    f = np.asarray(f, dtype=float)
    k0 = 2 * np.pi * f / c
    if model == 'delany_bazley':
        X = rho * f / sigma
        Z = 1 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732
        k = 1 + 0.0978 * X ** -0.700 - 1j * 0.189 * X ** -0.595
    elif model == 'miki':
        X = 1000 * f / sigma                      # f / sigma in Hz per kPa s/m^2
        Z = 1 + 5.50 * X ** -0.632 - 1j * 8.43 * X ** -0.632
        k = 1 + 7.81 * X ** -0.618 - 1j * 11.41 * X ** -0.618
    elif model == 'komatsu':
        u = 2 - np.log10(f / sigma)
        u = np.where(u >= 0, u, np.nan)
        Z = 1 + 0.00027 * u ** 6.2 - 1j * 0.0047 * u ** 4.1
        k = 1 + 0.0004 * u ** 6.2 - 1j * 0.0069 * u ** 4.1
    else:
        raise ValueError("model must be 'delany_bazley', 'miki' or 'komatsu'.")
    return rho * c * Z, k0 * k


def mpp_impedance(f, diameter, thickness, porosity, rho=_RHO, c=_C, eta=_ETA):
    """
    Series impedance (Pa s/m) of a microperforated panel, Maa (1998):

        Z / (rho c) = 32 eta t k_r / (p rho c d^2) + j w t k_m / (p c),
        k_r = sqrt(1 + x^2 / 32) + sqrt(2) x d / (32 t),
        k_m = 1 + 1 / sqrt(9 + x^2 / 2) + 0.85 d / t,  x = d sqrt(w rho / (4 eta)),

    with hole diameter d, panel thickness t and perforation ratio p.
    """
    f = np.asarray(f, dtype=float)
    w = 2 * np.pi * f
    d, t, p = diameter, thickness, porosity
    x = d * np.sqrt(w * rho / (4 * eta))
    kr = np.sqrt(1 + x ** 2 / 32) + np.sqrt(2) / 32 * x * d / t
    km = 1 + 1 / np.sqrt(9 + x ** 2 / 2) + 0.85 * d / t
    return rho * c * (32 * eta * t / (p * rho * c * d ** 2) * kr + 1j * w * t / (p * c) * km)


def _layer(f, layer, sin0, rho, c):
    """Transfer matrix (2, 2, nf) of one layer along the normal, for an
    incidence with sin(theta0) = sin0 in air (laterally reacting fluids)."""
    kind = layer[0]
    nf = np.size(f)
    if kind in ('air', 'porous'):
        if kind == 'air':
            Zc = np.full(nf, rho * c, dtype=complex)
            k = 2 * np.pi * np.asarray(f, dtype=float) / c + 0j
            d = layer[1]
        else:
            _, model, sigma, d = layer
            Zc, k = porous(f, sigma, model, rho, c)
        k0 = 2 * np.pi * np.asarray(f, dtype=float) / c
        kx = np.sqrt(k ** 2 - (k0 * sin0) ** 2)     # wavenumber along the normal
        Zn = Zc * k / kx
        cs, sn = np.cos(kx * d), np.sin(kx * d)
        return np.array([[cs, 1j * Zn * sn], [1j * sn / Zn, cs]])
    if kind == 'mpp':
        _, dia, t, p = layer
        Z = mpp_impedance(f, dia, t, p, rho, c)
    elif kind == 'mass':
        Z = 1j * 2 * np.pi * np.asarray(f, dtype=float) * layer[1]
    else:
        raise ValueError(f'unknown layer {kind!r}.')
    one, zero = np.ones(nf, dtype=complex), np.zeros(nf, dtype=complex)
    return np.array([[one, Z], [zero, one]])


def surface_impedance(f, layers, angle=0., rho=_RHO, c=_C):
    """
    Surface impedance (Pa s/m) of layers on a rigid backing, by the transfer
    matrix method, at an angle of incidence (degrees) in air.

    Parameters
    ----------
    f : array_like
        Frequencies (Hz).
    layers : list of tuples
        From the surface down to the backing:
        ('air', thickness), ('porous', model, sigma, thickness),
        ('mpp', hole diameter, thickness, perforation ratio) or
        ('mass', mass per unit area in kg/m^2).
    angle : float
        Angle of incidence in degrees. Air and porous layers react
        laterally (the trace wavenumber is kept); the panels react locally.
    """
    f = np.asarray(f, dtype=float)
    sin0 = np.sin(np.radians(angle))
    T = np.broadcast_to(np.eye(2, dtype=complex)[:, :, None], (2, 2, f.size)).copy()
    for layer in layers:
        T = np.einsum('ijn,jkn->ikn', T, _layer(f, layer, sin0, rho, c))
    with np.errstate(divide='ignore'):
        return T[0, 0] / T[1, 0]          # rigid backing: velocity 0 at the bottom


def absorption(Zs, angle=0., rho=_RHO, c=_C):
    """
    Absorption coefficient 1 - |R|^2 of a surface impedance at an angle of
    incidence (degrees), R = (Zs cos(theta) - rho c) / (Zs cos(theta) + rho c).
    """
    Zs = np.asarray(Zs, dtype=complex)
    a = Zs * np.cos(np.radians(angle))
    R = (a - rho * c) / (a + rho * c)
    return 1 - np.abs(R) ** 2


def random_incidence_absorption(f, layers=None, Zs=None, n_angles=180,
                                rho=_RHO, c=_C):
    """
    Random-incidence absorption by Paris' formula, the integral of
    alpha(theta) sin(2 theta) over 0 to 90 degrees.

    Give layers, whose surface impedance is computed at each angle, or the
    impedance Zs of a locally reacting surface, the same at all angles.
    """
    edges = np.linspace(0, 90, n_angles + 1)
    th = 0.5 * (edges[1:] + edges[:-1])
    dth = np.radians(edges[1] - edges[0])
    out = 0
    for a in th:
        Z = (surface_impedance(f, layers, a, rho, c) if Zs is None
             else np.broadcast_to(np.asarray(Zs, dtype=complex), np.shape(f)))
        out = out + absorption(Z, a, rho, c) * np.sin(2 * np.radians(a)) * dth
    return out
