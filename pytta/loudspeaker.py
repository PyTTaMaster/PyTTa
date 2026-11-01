"""
PyTTa loudspeaker:
------------------

Small signal (Thiele-Small) parameters of an electrodynamic loudspeaker from
its measured electrical impedance, with the lumped model of Thiele (1971)
and Small (1972):

    Z = Re + j w Le + Bl^2 / (Rms + j w Mms + 1 / (j w Cms)).

The free air curve gives Re, fs, Qms, Qes and Qts. A second curve, with a
known mass added to the diaphragm or with the driver on a closed box of
known volume, gives Mms, Cms, Rms, Bl and, with the effective diaphragm
area, Vas.

Available functions:

    >>> pytta.loudspeaker.thiele_small(impedance)
    >>> pytta.loudspeaker.impedance_model(params, freq)
    >>> pytta.loudspeaker.response(params, freq, area)
    >>> pytta.loudspeaker.max_spl(params, freq, area, xmax, power_max)
    >>> pytta.loudspeaker.radiation_impedance(freq, radius)
    >>> pytta.loudspeaker.closed_box(params, Vb)
    >>> pytta.loudspeaker.horn_profile(throat_area, cutoff, length)
    >>> pytta.loudspeaker.fit_with_velocity(freq, impedance, velocity)
    >>> pytta.loudspeaker.fit_voice_coil(freq, blocked, 'L2R2')

References:

    A. N. Thiele, "Loudspeakers in vented boxes: Parts I and II",
    J. Audio Eng. Soc. 19 (1971) 382-392 and 471-483.

    R. H. Small, "Direct-radiator loudspeaker system analysis",
    J. Audio Eng. Soc. 20 (1972) 383-395.

    R. H. Small, "Closed-box loudspeaker systems, Part I: Analysis",
    J. Audio Eng. Soc. 20 (1972) 798-808.

    L. L. Beranek and T. J. Mellow, "Acoustics: Sound Fields and
    Transducers", Academic Press, 2012, section 6.10.

For further information, check the function specific documentation.

"""

import numpy as np
from scipy.optimize import least_squares

from pytta.classes import SignalObj


def impedance_model(params, freq):
    """
    Electrical impedance of the lumped loudspeaker model.

        Z = Re + j 2 pi f Le + Re (Qms / Qes) / (1 + j Qms (f / fs - fs / f))

    The last term is the motional impedance Bl^2 / (Rms + j w Mms +
    1 / (j w Cms)) written with fs = 1 / (2 pi sqrt(Mms Cms)),
    Qms = 2 pi fs Mms / Rms and Qes = 2 pi fs Mms Re / Bl^2. At f = 0 it is
    zero and Z = Re.

    Parameters
    ----------
    params : dict
        'Re' (ohm), 'fs' (Hz), 'Qms', 'Qes' and optionally 'Le' (H,
        default 0). The dict returned by thiele_small fits here.
    freq : array_like
        Frequencies in Hz.

    Returns
    -------
    numpy.ndarray
        Complex impedance in ohm, same shape as freq.

    """
    f = np.asarray(freq, dtype=float)
    Re, fs, Qms = params['Re'], params['fs'], params['Qms']
    res = Re * Qms / params['Qes']
    zmot = res * f * fs / (f * fs + 1j * Qms * (f**2 - fs**2))
    return Re + 2j * np.pi * f * params.get('Le', 0.0) + zmot


def thiele_small(impedance, Re=None, area=None, added_mass=None,
                 impedance_with_mass=None, added_volume=None,
                 impedance_in_box=None, fit=False, c=343.2, rho=1.204):
    """
    Thiele-Small parameters from the electrical impedance of a loudspeaker.

    Impedance method (Small 1972; Beranek and Mellow 2012, eqs. 6.58 and
    6.59): fs and Rmax at the impedance peak, found by a parabola through
    its three bins (the peak is the local maximum of |Z| next to the
    positive to negative sign change of the reactance, so the rise of |Z|
    due to Le at high frequencies is skipped), and f1 < fs < f2 where |Z| = sqrt(Rmax Re),
    interpolated linearly between bins. With r0 = Rmax / Re,

        Qms = fs sqrt(r0) / (f2 - f1),  Qes = Qms / (r0 - 1),
        Qts = Qms Qes / (Qms + Qes).

    Added mass dm (eq. 6.66), with fs' the resonance of the loaded curve:

        Mms = dm / ((fs / fs')^2 - 1),  Cms = 1 / ((2 pi fs)^2 Mms).

    Added volume Vb (closed box, eq. 6.72), with fc and Qec from the curve
    in the box:

        Vas = Vb (fc Qec / (fs Qes) - 1),  Cms = Vas / (rho c^2 Sd^2).

    Then Rms = 2 pi fs Mms / Qms, Bl = sqrt(2 pi fs Mms Re / Qes) and
    Vas = rho c^2 Sd^2 Cms.

    The closed form neglects the voice coil inductance; with fit=True every
    curve is also fitted by least squares to impedance_model over the whole
    band (Re, Le, fs, Qms, Qes; Re only when it is not given), starting
    from the closed form. The relative complex error is minimized.

    Differences from ita_thiele_small (ITA-Toolbox): Re defaults to the
    minimum of the real part below the resonance instead of the real part
    at the lowest frequency; the peak and the side frequencies are interpolated instead
    of read at whole bins; the bandwidth is taken on |Z| at sqrt(Rmax Re)
    instead of on |Z - Re| at (Rmax - Re) / sqrt(2), which gives the same
    Qms when Le = 0.

    Parameters
    ----------
    impedance : SignalObj or tuple (freq, Z)
        Free air impedance in ohm, complex. A SignalObj must have one
        channel in freqSignal; bins at f <= 0 are ignored.
    Re : float, optional
        DC resistance of the voice coil in ohm, as measured with an
        ohmmeter. Default: minimum of the real part of the impedance
        below the resonance (in the model the real part is Re plus a
        positive term that vanishes at low frequencies).
    area : float, optional
        Effective diaphragm area Sd in m^2. Needed for Vas (added mass) or
        for the mechanical parameters (added volume).
    added_mass : float, optional
        Mass in kg added to the diaphragm for impedance_with_mass.
    impedance_with_mass : SignalObj or tuple (freq, Z), optional
        Impedance with added_mass on the diaphragm.
    added_volume : float, optional
        Net volume in m^3 of the closed box for impedance_in_box.
    impedance_in_box : SignalObj or tuple (freq, Z), optional
        Impedance of the driver on the closed box of added_volume.
    fit : bool, optional
        Refine the closed form by least squares and return Le. Default
        False.
    c : float, optional
        Speed of sound in m/s. Default 343.2 (air at 20 degC).
    rho : float, optional
        Density of air in kg/m^3. Default 1.204 (air at 20 degC).

    Returns
    -------
    dict
        'Re' (ohm), 'Le' (H, only with fit), 'fs' (Hz), 'Qms', 'Qes',
        'Qts'; with a second curve also 'Rms' (kg/s), 'Mms' (kg),
        'Cms' (m/N) and 'Bl' (T m), and 'Vas' (m^3) when it can be found.
        With added_volume and no area only 'Vas' is added.

    Raises
    ------
    ValueError
        If the reactance does not turn from positive to negative inside
        the band, the peak lies at the band edge, |Z| does not fall to
        sqrt(Rmax Re) on both sides of it, if a second curve comes without
        its mass or volume (or both methods are given), or if the added
        mass does not lower the resonance or the box does not raise it.

    """
    if (added_mass is None) != (impedance_with_mass is None):
        raise ValueError("added_mass and impedance_with_mass go together.")
    if (added_volume is None) != (impedance_in_box is None):
        raise ValueError("added_volume and impedance_in_box go together.")
    if added_mass is not None and added_volume is not None:
        raise ValueError("Give either the added mass or the added volume.")
    for name, value in (('area', area), ('added_mass', added_mass),
                        ('added_volume', added_volume), ('Re', Re)):
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be positive.")

    freq, z = _curve(impedance)
    ts = _estimate(freq, z, Re, fit)
    Re = ts['Re']
    Qms, Qes = ts['Qms'], ts['Qes']
    ws = 2 * np.pi * ts['fs']
    out = dict(ts, Qts=Qms * Qes / (Qms + Qes))

    Mms = Cms = None
    if added_mass is not None:
        loaded = _estimate(*_curve(impedance_with_mass), Re, fit)
        if not loaded['fs'] < ts['fs']:
            raise ValueError("The added mass must lower the resonance.")
        Mms = added_mass / ((ts['fs'] / loaded['fs'])**2 - 1)
        Cms = 1 / (ws**2 * Mms)
    elif added_volume is not None:
        boxed = _estimate(*_curve(impedance_in_box), Re, fit)
        if not boxed['fs'] > ts['fs']:
            raise ValueError("The closed box must raise the resonance.")
        out['Vas'] = added_volume * (boxed['fs'] * boxed['Qes']
                                     / (ts['fs'] * Qes) - 1)
        if area is not None:
            Cms = out['Vas'] / (rho * c**2 * area**2)
            Mms = 1 / (ws**2 * Cms)

    if Mms is not None:
        out.update(Rms=ws * Mms / Qms, Mms=Mms, Cms=Cms,
                   Bl=np.sqrt(ws * Mms * Re / Qes))
        if area is not None:
            out['Vas'] = rho * c**2 * area**2 * Cms
    return out


def _curve(impedance):
    """Frequencies above 0 Hz and complex impedance from either input."""
    if isinstance(impedance, SignalObj):
        if impedance.numChannels != 1:
            raise ValueError("The impedance SignalObj must have one channel.")
        freq, z = impedance.freqVector, impedance.freqSignal[:, 0]
    else:
        freq, z = impedance
    freq = np.asarray(freq, dtype=float)
    z = np.asarray(z, dtype=complex)
    if freq.ndim != 1 or freq.shape != z.shape:
        raise ValueError("freq and Z must be 1-D arrays of the same length.")
    keep = freq > 0
    freq, z = freq[keep], z[keep]
    if freq.size < 3 or np.any(np.diff(freq) <= 0):
        raise ValueError("freq must increase and hold at least 3 bins > 0 Hz.")
    return freq, z


def _estimate(freq, z, Re, fit):
    """Re, fs, Qms, Qes (and Le with fit) of one curve."""
    mag = np.abs(z)
    # Resonance: the + to - sign change of the reactance with the largest
    # |Z|, then uphill to the local maximum of |Z|.
    down = np.flatnonzero((z.imag[:-1] > 0) & (z.imag[1:] <= 0))
    if down.size == 0:
        raise ValueError("The reactance never turns from positive to "
                         "negative: no resonance inside the band.")
    down = down + (mag[down + 1] > mag[down])
    k = down[np.argmax(mag[down])]
    while 0 < k < mag.size - 1 and mag[k + 1] > mag[k]:
        k += 1
    while 0 < k < mag.size - 1 and mag[k - 1] > mag[k]:
        k -= 1
    if k == 0 or k == mag.size - 1:
        raise ValueError("The impedance peak lies at the band edge.")
    fit_re = Re is None
    if Re is None:
        Re = np.min(z.real[:k])
    a, b, c = np.polyfit(freq[k-1:k+2] - freq[k], mag[k-1:k+2], 2)
    fs = freq[k] - b / (2 * a)
    zmax = c - b**2 / (4 * a)
    level = np.sqrt(zmax * Re)
    below = np.flatnonzero(mag[:k] < level)
    above = k + np.flatnonzero(mag[k:] < level)
    if below.size == 0 or above.size == 0:
        raise ValueError("|Z| does not fall to sqrt(Rmax Re) on both sides "
                         "of the peak.")
    i, j = below[-1], above[0]
    f1 = np.interp(level, mag[i:i+2], freq[i:i+2])
    f2 = np.interp(level, mag[j-1:j+1][::-1], freq[j-1:j+1][::-1])
    r0 = zmax / Re
    Qms = fs * np.sqrt(r0) / (f2 - f1)
    ts = dict(Re=Re, fs=fs, Qms=Qms, Qes=Qms / (r0 - 1))
    if fit:
        ts = _fit(freq, z, ts, fit_re)
    return ts


def _fit(freq, z, ts, fit_re):
    """Least squares fit of impedance_model, Le included."""
    names = ['Re'] * fit_re + ['fs', 'Qms', 'Qes', 'Le']
    le0 = max(z[-1].imag / (2 * np.pi * freq[-1]), 0.0)
    x0 = [ts[n] for n in names[:-1]] + [le0]

    def residual(x):
        err = (impedance_model(dict(ts, **dict(zip(names, x))), freq) - z)
        err /= np.abs(z)
        return np.concatenate([err.real, err.imag])

    sol = least_squares(residual, x0, bounds=(0, np.inf), x_scale='jac',
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return dict(ts, **dict(zip(names, sol.x)))


def radiation_impedance(freq, radius, c=343.2, rho=1.204):
    """
    Mechanical radiation impedance in N s/m on one side of a rigid piston of
    the given radius (m) in an infinite baffle, S rho c (1 - J1(2ka) / (ka)
    + j H1(2ka) / (ka)), with J1 and H1 the Bessel and Struve functions
    (Beranek and Mellow 2012, eqs. 13.117 and 13.118).
    """
    from scipy.special import j1, struve
    x = 2 * 2 * np.pi * np.asarray(freq, dtype=float) / c * radius
    with np.errstate(invalid='ignore', divide='ignore'):
        z = 1 - 2 * j1(x) / x + 2j * struve(1, x) / x
    z = np.where(x == 0, 0, z)
    return np.pi * radius**2 * rho * c * z


def response(params, freq, area, voltage=2.83, distance=1., c=343.2,
             rho=1.204):
    """
    Small signal response of a driver in an infinite baffle to an rms
    voltage, from its Thiele-Small parameters ('Re', 'Bl', 'Mms', 'Cms',
    'Rms' and optionally 'Le', as thiele_small returns them with a second
    curve) and its effective area Sd (m^2), as ita_thielesmall2velocity.

    The diaphragm velocity is v = Bl U / ((Re + j w Le) Zm + Bl^2) with
    Zm = Rms + j w Mms + 1 / (j w Cms) (Mms includes the air load), the
    pressure on axis at distance r in the far field of the baffled piston is
    j w rho Sd v e^(-jkr) / (2 pi r), and the radiated power is
    |v|^2 Re(Zr) with Zr from radiation_impedance.

    Returns a dict with 'velocity' (m/s), 'displacement' (m), 'current'
    (A), 'pressure' (Pa), all rms and complex, 'SPL' (dB re 20 uPa),
    'power' (W) and 'efficiency', the power over voltage^2 / Re (Beranek
    and Mellow 2012, eq. 6.47 in the mid band).
    """
    f = np.asarray(freq, dtype=float)
    jw = 2j * np.pi * f
    Ze = params['Re'] + jw * params.get('Le', 0.0)
    Zm = params['Rms'] + jw * params['Mms'] + 1 / (jw * params['Cms'])
    Bl = params['Bl']
    v = Bl * voltage / (Ze * Zm + Bl**2)
    p = jw * rho * area * v * np.exp(-jw / c * distance) / (2 * np.pi * distance)
    power = np.abs(v)**2 * radiation_impedance(f, np.sqrt(area / np.pi), c, rho).real
    return {'velocity': v, 'displacement': v / jw,
            'current': voltage * Zm / (Ze * Zm + Bl**2), 'pressure': p,
            'SPL': 20 * np.log10(np.abs(p) / 2e-5), 'power': power,
            'efficiency': power / (voltage**2 / params['Re'])}


def max_spl(params, freq, area, xmax, power_max, distance=1., c=343.2,
            rho=1.204):
    """
    Largest SPL at each frequency limited by the peak excursion xmax (m) and
    by the electrical power power_max (W) in Re, as
    ita_loudspeakertools_maxSPL. Returns (SPL, rms voltage, limit), limit
    being 'excursion' or 'thermal' at each frequency.
    """
    one = response(params, freq, area, 1., distance, c, rho)
    u_x = xmax / (np.sqrt(2) * np.abs(one['displacement']))
    u_t = np.sqrt(power_max / (np.abs(one['current'])**2 * params['Re']))
    u = np.minimum(u_x, u_t)
    return one['SPL'] + 20 * np.log10(u), u, np.where(u_x < u_t, 'excursion', 'thermal')


def closed_box(params, Vb):
    """
    Resonance frequency and total Q of the driver in a closed box of net
    volume Vb (m^3), fc = fs sqrt(1 + Vas / Vb) and Qtc = Qts sqrt(1 +
    Vas / Vb), without box losses (Small 1972, closed box; Beranek and
    Mellow 2012, eq. 7.28).
    """
    k = np.sqrt(1 + params['Vas'] / Vb)
    return params['fs'] * k, params['Qts'] * k


def horn_profile(throat_area, cutoff, length, a=1., c=343.2, num=200):
    """
    Cross-sectional area along a hyperbolic-exponential (hypex) horn,
    S = St (cosh(x / xT) + a sinh(x / xT))^2 with xT = c / (2 pi fc), from
    hyperbolic (a = 0) to exponential (a = 1, S = St e^(mx), m = 4 pi fc /
    c) (Beranek and Mellow 2012, eqs. 9.39, 9.51 and 9.58). Returns
    (x, S) at num points from the throat to length (m).
    """
    x = np.linspace(0, length, num)
    xT = c / (2 * np.pi * cutoff)
    return x, throat_area * (np.cosh(x / xT) + a * np.sinh(x / xT))**2


def fit_with_velocity(freq, impedance, velocity, weights=None):
    """
    Thiele-Small parameters from the electrical impedance Z = U / i and the
    diaphragm velocity per volt H = v / U measured together (e.g. with a
    laser vibrometer), as ita_thiele_small_fit and, with weights per
    frequency, ita_thiele_small_fit_weighted. With G = v / i = H Z, the
    model gives two linear least squares problems:

        Z = Re + j w Le + Bl G,
        Bl / G = Rms + j w Mms + 1 / (j w Cms).

    Returns a dict with 'Re', 'Le', 'Bl', 'Rms', 'Mms', 'Cms', 'fs', 'Qms',
    'Qes' and 'Qts'.
    """
    w = 2 * np.pi * np.asarray(freq, dtype=float)
    Z = np.asarray(impedance)
    G = np.asarray(velocity) * Z
    sw = np.sqrt(np.ones(w.size) if weights is None else np.asarray(weights))

    def solve(columns, target):
        M = np.column_stack(columns) * sw[:, None]
        rhs = target * sw
        return np.linalg.lstsq(np.vstack([M.real, M.imag]),
                               np.r_[rhs.real, rhs.imag], rcond=None)[0]

    Re, Le, Bl = solve([np.ones_like(w), 1j * w, G], Z)
    Rms, Mms, K = solve([np.ones_like(w), 1j * w, 1 / (1j * w)], Bl / G)
    ws = np.sqrt(K / Mms)
    Qms, Qes = ws * Mms / Rms, ws * Mms * Re / Bl**2
    return {'Re': Re, 'Le': Le, 'Bl': Bl, 'Rms': Rms, 'Mms': Mms,
            'Cms': 1 / K, 'fs': ws / (2 * np.pi), 'Qms': Qms, 'Qes': Qes,
            'Qts': Qms * Qes / (Qms + Qes)}


def voice_coil_model(params, freq):
    """
    Blocked electrical impedance of the voice coil: Re + K (j w)^n when
    params has 'K' and 'n' (Leach, Loudspeaker voice-coil inductance
    losses, J. Audio Eng. Soc. 50(6), 2002), otherwise Re + j w Le plus,
    with 'L2' and 'R2', the eddy current branch j w L2 R2 / (R2 + j w L2)
    (the L2R2 model).
    """
    jw = 2j * np.pi * np.asarray(freq, dtype=float)
    if 'K' in params:
        return params['Re'] + params['K'] * jw ** params['n']
    z = params['Re'] + jw * params.get('Le', 0.)
    if 'L2' in params:
        z = z + jw * params['L2'] * params['R2'] / (params['R2'] + jw * params['L2'])
    return z


def fit_voice_coil(freq, blocked, model='L2R2', Re=None):
    """
    Parameters of voice_coil_model fitted by least squares on the relative
    complex error to a blocked impedance (for instance Z - Bl G from
    fit_with_velocity, or a measurement with the diaphragm held), as
    ita_inductance_fit and ita_curvefit_inductance. model is 'Le', 'L2R2'
    or 'Leach'; Re is fitted unless given.
    """
    f = np.asarray(freq, dtype=float)
    z = np.asarray(blocked)
    names = {'Le': ['Le'], 'L2R2': ['Le', 'L2', 'R2'], 'Leach': ['K', 'n']}[model]
    w = 2 * np.pi * f
    Le0 = np.median(z.imag / w)
    x0 = {'Le': Le0, 'L2': Le0, 'R2': z.real.max(), 'K': Le0, 'n': 0.9}
    fit_re = Re is None
    start = [x0[n] for n in names] + ([z.real.min()] if fit_re else [])

    def params(x):
        p = dict(zip(names, x))
        p['Re'] = x[-1] if fit_re else Re
        return p

    def residual(x):
        err = voice_coil_model(params(x), f) / z - 1
        return np.concatenate([err.real, err.imag])

    sol = least_squares(residual, start, bounds=(0, np.inf), x_scale='jac',
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return params(sol.x)
