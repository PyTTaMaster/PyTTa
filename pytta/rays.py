# -*- coding: utf-8 -*-
"""
Stochastic ray tracing in convex rooms (Krokstad, Strom and Sorsdal,
Calculating the acoustical room response by the use of a ray tracing
technique, J. Sound Vib. 8(1), 1968; Vorlaender, Auralization, 2008),
as in the ray tracing part of RAVEN, which the ITA-Toolbox drives.

A room is a list of planes, each with its absorption and scattering
coefficients. Rays leave the source in random directions with equal
energy, lose the absorbed share at each wall and are reflected
specularly, or, with the probability of the scattering coefficient, in a
random direction drawn from Lambert's law. A sphere around the receiver
counts each ray crossing it with its energy times the chord length over
the sphere volume, so the histogram is the energy per unit area arriving
in each time bin (E0 / (4 pi r^2) for the direct sound).

>>> room = pytta.rays.shoebox(6, 5, 4, alpha=0.2, scattering=0.5)
>>> t, h = pytta.rays.trace(room, [1, 1, 1.5], [4, 3, 1.2], duration=1.)

"""

import numpy as np

__all__ = ['shoebox', 'convex_room', 'trace']


def shoebox(Lx, Ly, Lz, alpha, scattering=0.):
    """Room [0, Lx] x [0, Ly] x [0, Lz]; alpha and scattering are one value
    for all walls or six, in the order x = 0, x = Lx, y = 0, y = Ly, z = 0,
    z = Lz."""
    normals = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                        [0, 0, -1], [0, 0, 1]], float)
    offsets = np.array([0, Lx, 0, Ly, 0, Lz], float)
    return _room(normals, offsets, alpha, scattering)


def convex_room(vertices, alpha, scattering=0.):
    """Convex room spanned by its corner points (M, 3), one plane per face of
    their convex hull; alpha and scattering are one value for all faces or
    one per face, in the order of room['normals']."""
    from scipy.spatial import ConvexHull
    eq = ConvexHull(np.asarray(vertices, float)).equations
    normals, offsets = eq[:, :3], -eq[:, 3]
    keep = np.unique(np.round(np.column_stack([normals, offsets]), 9), axis=0, return_index=True)[1]
    keep.sort()
    return _room(normals[keep], offsets[keep], alpha, scattering)


def _room(normals, offsets, alpha, scattering):
    n = len(offsets)
    return {'normals': normals, 'offsets': offsets,
            'alpha': np.broadcast_to(np.asarray(alpha, float), (n,)).copy(),
            'scattering': np.broadcast_to(np.asarray(scattering, float), (n,)).copy()}


def _lambert(normals, rng):
    # Directions about the inward normals -normals with density cos(theta).
    u, v = rng.uniform(size=(2, len(normals)))
    z = np.sqrt(1 - u)
    r = np.sqrt(u)
    phi = 2 * np.pi * v
    axis = -normals
    helper = np.where(np.abs(axis[:, :1]) < 0.9, [[1., 0, 0]], [[0, 1., 0]])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axis, e1)
    return (r * np.cos(phi))[:, None] * e1 + (r * np.sin(phi))[:, None] * e2 + z[:, None] * axis


def trace(room, source, receiver, numRays=20000, duration=1., radius=0.5,
          dt=1e-3, c=343., m=0., seed=None):
    """
    Energy histogram at the receiver of a unit energy impulse at the
    source: returns (bin start times in s, energy per unit area per bin).
    m is the power attenuation of air in 1/m.
    """
    rng = np.random.default_rng(seed)
    src, rcv = np.asarray(source, float), np.asarray(receiver, float)
    D = rng.standard_normal((numRays, 3))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    P = np.tile(src, (numRays, 1))
    E = np.full(numRays, 1 / numRays)
    L = np.zeros(numRays)
    hist = np.zeros(int(np.ceil(duration / dt)))
    volume = 4 / 3 * np.pi * radius**3
    nrm, off = room['normals'], room['offsets']
    alive = np.ones(numRays, bool)
    while alive.any():
        P, D, E, L = P[alive], D[alive], E[alive], L[alive]
        # Distance to the wall the ray leaves through.
        speed = D @ nrm.T
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(speed > 1e-12, (off - P @ nrm.T) / speed, np.inf)
        wall = np.argmin(t, axis=1)
        t = t[np.arange(len(t)), wall]
        # Crossing of the receiver sphere on the way.
        s = np.sum((rcv - P) * D, axis=1)
        d2 = np.sum((rcv - P)**2, axis=1) - s**2
        hit = (s > 0) & (s < t) & (d2 < radius**2)
        when = ((L + s)[hit] / c / dt).astype(int)
        ok = when < hist.size
        chord = 2 * np.sqrt(radius**2 - d2[hit])
        np.add.at(hist, when[ok], (E[hit] * chord / volume * np.exp(-m * (L + s)[hit]))[ok])
        # Reflection.
        P = P + t[:, None] * D
        L = L + t
        E = E * (1 - room['alpha'][wall]) * np.exp(-m * t)
        n = nrm[wall]
        specular = D - 2 * np.sum(D * n, axis=1, keepdims=True) * n
        diffuse = rng.uniform(size=len(D)) < room['scattering'][wall]
        D = np.where(diffuse[:, None], _lambert(n, rng), specular)
        alive = (L < c * duration) & (E > 0)
    return np.arange(hist.size) * dt, hist
