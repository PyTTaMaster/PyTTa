"""
PyTTa VBAP:
-----------

Vector base amplitude panning (VBAP) of Pulkki (1997) for 2D and 3D
loudspeaker layouts, and its spread version, multiple-direction amplitude
panning (MDAP).

Available functions:

    >>> groups = pytta.vbap.loudspeaker_pairs(azimuths)        # 2D
    >>> groups = pytta.vbap.loudspeaker_triplets(directions)   # 3D
    >>> inverses = pytta.vbap.inverse_matrices(directions, groups)
    >>> G = pytta.vbap.gains(source_directions, groups, inverses, spread=0)
    >>> G = pytta.vbap.energy_gains(source_directions, groups, inverses)

Directions are in degrees: azimuth for 2D layouts, [azimuth, elevation] for
3D ones, with x = cos(azi) cos(elev), y = sin(azi) cos(elev) and z =
sin(elev) (0 deg in front, 90 deg to the left). Loudspeaker indices are
0-based.

Origin of the code
------------------

This module is a port to Python of the MATLAB library
Vector-Base-Amplitude-Panning by Archontis Politis and Ville Pulkki (2015),
https://github.com/polarch/Vector-Base-Amplitude-Panning (findLsPairs.m,
findLsTriplets.m, invertLsMtx.m, vbap.m, vbip.m, getSpreadSrcDirs.m), under the BSD
3-Clause licence reproduced below. The gains follow the MATLAB code
operation by operation.

References
----------

Pulkki, V. (1997). Virtual sound source positioning using vector base
amplitude panning. Journal of the Audio Engineering Society, 45(6), 456-466.

Pulkki, V. (1999). Uniform spreading of amplitude panned virtual sources.
Proceedings of the IEEE Workshop on Applications of Signal Processing to
Audio and Acoustics, New Paltz, NY.

Licence of the ported code
--------------------------

    Copyright (c) 2015, Archontis Politis
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.spatial import ConvexHull


def _cartesian(directions):
    d = np.radians(np.asarray(directions, dtype=float))
    if d.ndim == 1:
        return np.column_stack([np.cos(d), np.sin(d)])
    azi, elev = d[:, 0], d[:, 1]
    return np.column_stack([np.cos(azi) * np.cos(elev),
                            np.sin(azi) * np.cos(elev), np.sin(elev)])


def loudspeaker_pairs(azimuths, max_aperture=None):
    """
    Adjacent loudspeaker pairs of a 2D layout (findLsPairs).

    Parameters
    ----------
    azimuths : array_like
        Loudspeaker azimuths in degrees.
    max_aperture : float, optional
        Pairs that span this angle in degrees or more are left out.

    Returns
    -------
    numpy.ndarray
        (pairs, 2) loudspeaker indices, around the circle by azimuth.
    """
    azimuths = np.asarray(azimuths, dtype=float)
    order = np.argsort(azimuths, kind='stable')
    groups = np.column_stack([order, np.roll(order, -1)])
    if max_aperture is not None:
        s = azimuths[np.append(order, order[0])]
        apertures = np.diff(s)
        apertures[-1] = 360 - abs(apertures[-1])
        groups = groups[apertures < max_aperture]
    return groups


def loudspeaker_triplets(directions, max_aperture=None):
    """
    Loudspeaker triplets of a 3D layout (findLsTriplets).

    The triplets are the faces of the convex hull of the loudspeaker
    directions, which on the unit sphere is their Delaunay triangulation.
    Faces whose outward normal is at 90 deg or more from their centroid are
    left out: those are the faces that close a layout with no loudspeakers
    below or above, through which no source direction passes.

    Parameters
    ----------
    directions : array_like
        (loudspeakers, 2) [azimuth, elevation] in degrees.
    max_aperture : float, optional
        Triplets with a side longer than this angle in degrees are left out.

    Returns
    -------
    numpy.ndarray
        (triplets, 3) loudspeaker indices, each triplet counter-clockwise
        seen from outside and starting at its smallest index, sorted.
    """
    U = _cartesian(directions)
    faces = []
    for face in ConvexHull(U).simplices:
        v = U[face]
        n = np.cross(v[1] - v[0], v[2] - v[1])
        if np.dot(n, v.mean(axis=0) - U.mean(axis=0)) < 0:
            face, v, n = face[::-1], v[::-1], -n
        # MATLAB keeps acos(dot(centroid, n)) < pi/2; a dot product of
        # rounding size, as on a flat bottom face, rounds to pi/2 there.
        if np.dot(v.mean(axis=0), n) > 1e-12:
            faces.append(np.roll(face, -int(np.argmin(face))))
    faces = np.array(faces, dtype=int)
    if max_aperture is not None:
        sides = np.stack([np.sum(U[faces[:, a]] * U[faces[:, b]], axis=1)
                          for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
        faces = faces[np.all(np.arccos(np.clip(sides, -1, 1))
                             <= np.radians(max_aperture), axis=1)]
    return faces[np.lexsort((faces[:, 1], faces[:, 0]))]


def inverse_matrices(directions, groups):
    """
    Inverse of the matrix of loudspeaker vectors of each group (invertLsMtx).

    Parameters
    ----------
    directions : array_like
        Loudspeaker azimuths (2D) or [azimuth, elevation] (3D) in degrees.
    groups : numpy.ndarray
        Pairs or triplets from loudspeaker_pairs or loudspeaker_triplets.

    Returns
    -------
    numpy.ndarray
        (groups, dim, dim): the inverse of the matrix whose rows are the
        unit vectors of the group, so that the gains of a source u are
        u @ inverse.
    """
    U = _cartesian(directions)
    return np.linalg.inv(U[np.asarray(groups)])


def _spread_directions(direction, spread, num_src=8, num_rings=1):
    # getSpreadSrcDirs: num_src sources across the spread in 2D; in 3D
    # num_rings concentric rings of num_src sources and the source itself.
    if np.ndim(direction) == 0:
        d = np.radians(np.linspace(direction - spread / 2, direction + spread / 2,
                                   num_src + 1))
        return np.column_stack([np.cos(d), np.sin(d)])
    u = _cartesian(np.atleast_2d(direction))[0]
    u_x = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    theta = 2 * np.pi / num_src
    R = (np.eye(3) * np.cos(theta) + np.sin(theta) * u_x
         + (1 - np.cos(theta)) * np.outer(u, u))
    base = np.zeros((num_src, 3))
    if direction[1] > 89 or direction[1] < -89:
        base[0] = [1, 0, 0]
    else:
        c = np.cross(u, [0, 0, 1])
        base[0] = c / np.sqrt(np.sum(c**2))
    for n in range(1, num_src):
        base[n] = R @ base[n - 1]
    ring = np.radians(spread / 2) / num_rings
    rings = np.vstack([u + base * np.tan(ring * (r + 1)) for r in range(num_rings)])
    rings = rings / np.sqrt(np.sum(rings**2, axis=1))[:, None]
    return np.vstack([rings, u])


def _group_gains(u, groups, inverses, first_only, energy=False):
    # One direction: the gains of the first (3D) or last (2D) group that
    # holds it, normalized in amplitude (VBAP) or in energy (VBIP).
    g = np.zeros(int(groups.max()) + 1)
    for group, inv in zip(groups, inverses):
        g_tmp = u @ inv
        if np.min(g_tmp) > -0.001:
            g[group] = g_tmp / (np.sum(g_tmp) if energy else np.sqrt(np.sum(g_tmp**2)))
            if first_only:
                break
    return g


def gains(directions, groups, inverses, spread=0.):
    """
    VBAP gains of a set of source directions (vbap).

    Parameters
    ----------
    directions : array_like
        Source azimuths (2D) or (sources, 2) [azimuth, elevation] (3D) in
        degrees.
    groups : numpy.ndarray
        Pairs or triplets from loudspeaker_pairs or loudspeaker_triplets.
    inverses : numpy.ndarray
        From inverse_matrices.
    spread : float, optional
        Spread in degrees for MDAP: the gains of 9 directions across the
        spread (8 on a ring and the source in 3D) are summed. The default is
        0, plain VBAP.

    Returns
    -------
    numpy.ndarray
        (sources, loudspeakers) gains, non-negative, of norm 1 per source.
        A source outside every group (below a hemisphere layout, say) gets
        all gains 0, as in the MATLAB code.
    """
    groups = np.asarray(groups)
    three_d = groups.shape[1] == 3
    directions = np.asarray(directions, dtype=float)
    directions = directions.reshape(-1, 2) if three_d else directions.reshape(-1)
    out = np.zeros((len(directions), int(groups.max()) + 1))
    for n, d in enumerate(directions):
        if spread:
            g = sum(_group_gains(u, groups, inverses, three_d)
                    for u in _spread_directions(d, spread))
        else:
            g = _group_gains(_cartesian(np.atleast_2d(d) if three_d else [d])[0],
                             groups, inverses, three_d)
        with np.errstate(invalid='ignore'):
            out[n] = np.fmax(0, g / np.sqrt(np.sum(g**2)))
    return out


def energy_gains(directions, groups, inverses, spread=0., num_src=8,
                 num_rings=1):
    """
    Vector base intensity panning gains (vbip).

    The energies of the loudspeakers of a group, not their amplitudes,
    weight the loudspeaker vectors into the source direction; the gains are
    their square roots, normalized in energy.

    Parameters
    ----------
    directions, groups, inverses
        As in gains.
    spread : float, optional
        Spread in degrees: the energies of num_src directions across it (in
        3D num_rings rings of num_src and the source) are summed.
    num_src, num_rings : int, optional
        Spread directions, 8 and 1 by default.

    Returns
    -------
    numpy.ndarray
        (sources, loudspeakers) gains, of norm 1 per source.
    """
    groups = np.asarray(groups)
    three_d = groups.shape[1] == 3
    directions = np.asarray(directions, dtype=float)
    directions = directions.reshape(-1, 2) if three_d else directions.reshape(-1)
    out = np.zeros((len(directions), int(groups.max()) + 1))
    for n, d in enumerate(directions):
        if spread:
            e = sum(_group_gains(u, groups, inverses, three_d, energy=True)
                    for u in _spread_directions(d, spread, num_src, num_rings))
        else:
            e = _group_gains(_cartesian(np.atleast_2d(d) if three_d else [d])[0],
                             groups, inverses, three_d, energy=True)
        with np.errstate(invalid='ignore'):
            e = e / np.sum(e)
        if three_d:
            e[e < 0] = 0
        out[n] = np.fmax(0, np.sqrt(e))
    return out
