# -*- coding: utf-8 -*-
"""
Boundary element method for sound radiated by a vibrating body into free
field, by collocation with constant triangular elements (Ciskowski and
Brebbia, Boundary Element Methods in Acoustics, 1991; Marburg and Nolte,
Computational Acoustics of Noise Propagation in Fluids, 2008). On the
surface,

    1/2 p(x) = int_S [p(y) dG/dn(x, y) - G(x, y) q(y)] dS(y),

with G = exp(-jkr) / (4 pi r) (time dependence exp(j w t)), n the normal
pointing into the fluid and q = dp/dn = -j w rho v_n for the normal
velocity v_n of the surface into the fluid. CHIEF points inside the body
(Schenck, J. Acoust. Soc. Am. 44, 1968), where the same integral must
vanish, remove the irregular frequencies by least squares.

>>> nodes, tris = pytta.bem.icosphere(0.1, 3)
>>> p = pytta.bem.surface_pressure(nodes, tris, 500., 1e-3)
>>> pf = pytta.bem.field(nodes, tris, 500., p, 1e-3, [[0, 0, 1.]])

"""

import numpy as np

__all__ = ['icosphere', 'surface_pressure', 'field']

# Seven-point rule on the triangle (Dunavant 1985, degree 5): barycentric
# coordinates and weights summing to 1.
_A, _B = 0.059715871789770, 0.470142064105115
_C, _D = 0.797426985353087, 0.101286507323456
_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A],
                  [_C, _D, _D], [_D, _C, _D], [_D, _D, _C]])
_W = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def icosphere(radius=1., level=3):
    """Triangulated sphere: an icosahedron with every triangle split in
    four level times, the nodes pushed onto the sphere. Returns (nodes,
    triangles) with the triangles ordered so that their normals point
    out."""
    t = (1 + np.sqrt(5)) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
         [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
         [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    nodes = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, new = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = nodes[a] + nodes[b]
                nodes.append(m / np.linalg.norm(m))
                cache[key] = len(nodes) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = new
    return radius * np.array(nodes), np.array(f)


def _geometry(nodes, tris):
    P = nodes[tris]
    cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    return P, P.mean(axis=1), cross / (2 * area[:, None]), area


def _kernels(x, y, normal, k):
    # G and dG/dn_y between points x (m, 3) and y (..., 3).
    d = y - x
    r = np.linalg.norm(d, axis=-1)
    G = np.exp(-1j * k * r) / (4 * np.pi * r)
    dG = -G * (1 + 1j * k * r) / r**2 * np.sum(d * normal, axis=-1)
    return G, dG


def _matrices(nodes, tris, x, k, on_surface):
    # Gm[i, j] = int_Tj G(x_i, y) dS, Hm[i, j] = int_Tj dG/dn(x_i, y) dS.
    P, centre, normal, area = _geometry(nodes, tris)
    y = np.einsum('qa,tab->tqb', _BARY, P)                      # (T, 7, 3)
    with np.errstate(divide='ignore', invalid='ignore'):     # own centroid
        G, dG = _kernels(x[:, None, None, :], y[None], normal[None, :, None, :], k)
    Gm = np.einsum('itq,q->it', G, _W) * area
    Hm = np.einsum('itq,q->it', dG, _W) * area
    if on_surface:
        for i in range(x.shape[0]):
            Gm[i, i] = _self_g(P[i], centre[i], k)
            Hm[i, i] = 0.                           # flat element: d.n = 0
    return Gm, Hm


def _self_g(tri, x, k):
    # int G over the element seen from its centroid: the 1 / r part in polar
    # coordinates over the three sub-triangles, the rest by the rule.
    total = 0.
    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        e = b - a
        h_vec = (a - x) - np.dot(a - x, e) / np.dot(e, e) * e
        h = np.linalg.norm(h_vec)
        u = h_vec / h
        v = e / np.linalg.norm(e)
        t1, t2 = (np.arctan2(np.dot(p - x, v), np.dot(p - x, u)) for p in (a, b))
        theta = (t2 - t1) / 2 * _GL_X + (t2 + t1) / 2
        total += (t2 - t1) / 2 * np.sum(_GL_W * h / np.cos(theta))
    y = _BARY @ tri
    r = np.linalg.norm(y - x, axis=1)
    area = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(r > 0, (np.exp(-1j * k * r) - 1) / r, -1j * k)
    regular = np.sum(_W * f) * area
    return (total + regular) / (4 * np.pi)


def surface_pressure(nodes, tris, freq, velocity, chief=None, c=343., rho=1.21):
    """
    Sound pressure on each element of a closed surface (normals pointing
    into the fluid) vibrating with normal velocity velocity (m/s, one value
    per element or one for all) at freq (Hz). chief: points inside the body
    (CHIEF), by default ten random points within half the distance from
    the centroid of the body to its surface.
    """
    k = 2 * np.pi * freq / c
    _, centre, _, _ = _geometry(nodes, tris)
    q = -1j * 2 * np.pi * freq * rho * np.broadcast_to(velocity, (tris.shape[0],))
    Gm, Hm = _matrices(nodes, tris, centre, k, True)
    A, b = 0.5 * np.eye(tris.shape[0]) - Hm, -Gm @ q
    if chief is None:
        rng = np.random.default_rng(0)
        mid = nodes.mean(axis=0)
        reach = 0.5 * np.min(np.linalg.norm(centre - mid, axis=1))
        chief = mid + reach * rng.uniform(-1, 1, (10, 3)) / np.sqrt(3)
    chief = np.atleast_2d(chief)
    if chief.size:
        Gc, Hc = _matrices(nodes, tris, chief, k, False)
        A, b = np.vstack([A, -Hc]), np.concatenate([b, -Gc @ q])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def field(nodes, tris, freq, pressure, velocity, points, c=343., rho=1.21):
    """Sound pressure at points (M, 3) in the fluid from the surface pressure
    of surface_pressure and the normal velocity of the elements."""
    k = 2 * np.pi * freq / c
    q = -1j * 2 * np.pi * freq * rho * np.broadcast_to(velocity, (tris.shape[0],))
    Gm, Hm = _matrices(nodes, tris, np.atleast_2d(np.asarray(points, float)), k, False)
    return Hm @ pressure - Gm @ q
