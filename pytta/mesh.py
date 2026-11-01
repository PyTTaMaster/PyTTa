# -*- coding: utf-8 -*-
"""
Meshes for pytta.fem and pytta.bem: tetrahedra (or triangles in 2D) from a
point cloud, quality meshes of a domain given by its signed distance
function (DistMesh), meshes read from other programs, and the boundary
surface of a volume mesh.

>>> nodes, tets = pytta.mesh.distmesh(pytta.mesh.sphere([0, 0, 0], 1.), 0.2,
...                                   [[-1, -1, -1], [1, 1, 1]])
>>> surface = pytta.mesh.boundary(nodes, tets)     # triangles for pytta.bem

"""

import warnings

import numpy as np
from scipy.spatial import Delaunay

__all__ = ['sphere', 'box', 'union', 'difference', 'intersection',
           'delaunay_mesh', 'distmesh', 'boundary', 'read_mesh', 'quality']


# Signed distance functions: negative inside, zero on the boundary.

def sphere(center, radius):
    """Distance to a sphere (a circle in 2D)."""
    c = np.asarray(center, float)
    return lambda p: np.linalg.norm(p - c, axis=1) - radius


def box(lower, upper):
    """Distance to an axis-aligned box (a rectangle in 2D), exact outside
    and inside."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)

    def d(p):
        q = np.maximum(lo - p, p - hi)
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        return outside + np.minimum(q.max(axis=1), 0)
    return d


def union(*fds):
    return lambda p: np.min([f(p) for f in fds], axis=0)


def intersection(*fds):
    return lambda p: np.max([f(p) for f in fds], axis=0)


def difference(fd1, fd2):
    """Points of fd1 that are not in fd2."""
    return lambda p: np.maximum(fd1(p), -fd2(p))


def delaunay_mesh(points, fd=None, tol=0.):
    """
    Simplices (tetrahedra in 3D, triangles in 2D) of the Delaunay
    triangulation of points (N, dim). It fills the convex hull of the
    points; for a domain that is not convex, give its signed distance fd
    and the simplices whose centroid has fd >= -tol are dropped.

    Points in degenerate position (four or more on a sphere in 3D, as on a
    regular grid) make Qhull return simplices of zero volume. Those on the
    convex hull are dropped, which leaves a valid mesh; one inside would
    leave two faces that do not match, so a warning asks to perturb the
    points or to use a structured mesh such as pytta.fem.box_mesh.
    """
    p = np.asarray(points, float)
    tri = Delaunay(p)
    t = tri.simplices
    size = np.abs(np.linalg.det(p[t[:, 1:]] - p[t[:, :1]]))
    flat = size <= 1e-10 * size.mean()
    on_hull = np.zeros(len(p), bool)
    on_hull[np.unique(tri.convex_hull)] = True
    if np.any(flat & ~on_hull[t].all(axis=1)):
        warnings.warn('delaunay_mesh: flat simplices inside the mesh; perturb '
                      'the points or use a structured mesh.')
    t = t[~flat]
    if fd is not None:
        t = t[fd(p[t].mean(axis=1)) < -tol]
    return t


def distmesh(fd, h0, bbox, fixed=None, fh=None, maxiter=500, seed=0):
    """
    Mesh of the domain fd < 0 by the algorithm of Persson and Strang (A
    simple mesh generator in MATLAB, SIAM Review 46(2), 2004), written from
    the paper. Points on a regular grid of spacing h0 in bbox [[min...],
    [max...]] (equilateral rows in 2D, thinned by the relative size fh if
    given) move as the nodes of a truss whose bars push apart when shorter
    than their target length; the Delaunay triangulation is redone when a
    point has moved more than ttol h0, and points that leave the domain are
    projected back by p - d grad d. fixed points never move. The loop stops
    when no interior point moves more than dptol h0, or after maxiter
    steps. In 2D the parameters are those of the paper (Fscale 1.2, time
    step 0.2, geps 1e-3 h0, ttol 0.1, dptol 1e-3); in 3D the paper gives
    no values, and Fscale 1.1 with a time step of 0.1 are our choice. As
    the paper notes, Delaunay meshes in 3D may hold slivers (flat
    tetrahedra of good edge lengths); quality() finds them. Returns
    (nodes, simplices); dim follows bbox.
    """
    bbox = np.asarray(bbox, float)
    dim = bbox.shape[1]
    geps, deps, dptol, ttol = 1e-3 * h0, np.sqrt(np.finfo(float).eps) * h0, 1e-3, 0.1
    deltat, fscale = (0.2, 1.2) if dim == 2 else (0.1, 1 + 0.4 / 2 ** (dim - 1))
    fh = fh or (lambda p: np.ones(len(p)))
    axes = [np.arange(bbox[0, i], bbox[1, i] + h0 / 2, h0) for i in range(dim)]
    if dim == 2:      # equilateral rows
        x, y = np.meshgrid(axes[0], np.arange(bbox[0, 1], bbox[1, 1] + h0, h0 * np.sqrt(3) / 2))
        x[1::2] += h0 / 2
        p = np.column_stack([x.ravel(), y.ravel()])
    else:
        p = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, dim)
    p = p[fd(p) < geps]
    r0 = 1 / fh(p) ** dim
    p = p[np.random.default_rng(seed).uniform(size=len(p)) < r0 / r0.max()]
    pfix = np.zeros((0, dim)) if fixed is None else np.asarray(fixed, float).reshape(-1, dim)
    p = np.vstack([pfix, p])
    nfix = len(pfix)
    old = np.inf * np.ones_like(p)
    for _ in range(maxiter):
        if np.max(np.linalg.norm(p - old, axis=1)) / h0 > ttol:
            old = p.copy()
            with warnings.catch_warnings():     # the first grid is degenerate
                warnings.simplefilter('ignore')
                t = delaunay_mesh(p, fd, geps)
            pairs = np.array([[i, j] for i in range(dim + 1) for j in range(i + 1, dim + 1)])
            bars = np.unique(np.sort(t[:, pairs].reshape(-1, 2), axis=1), axis=0)
        vec = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.linalg.norm(vec, axis=1)
        hbars = fh((p[bars[:, 0]] + p[bars[:, 1]]) / 2)
        L0 = hbars * fscale * (np.sum(L ** dim) / np.sum(hbars ** dim)) ** (1 / dim)
        F = np.maximum(L0 - L, 0)
        Fvec = (F / L)[:, None] * vec
        Ftot = np.zeros_like(p)
        np.add.at(Ftot, bars[:, 0], Fvec)
        np.add.at(Ftot, bars[:, 1], -Fvec)
        Ftot[:nfix] = 0
        p = p + deltat * Ftot
        d = fd(p)
        out = d > 0
        if out.any():
            grad = np.column_stack([(fd(p[out] + deps * np.eye(dim)[i]) - d[out]) / deps
                                    for i in range(dim)])
            p[out] -= d[out][:, None] * grad
        inside = d < -geps
        if np.max(np.linalg.norm(deltat * Ftot[inside], axis=1), initial=0) / h0 < dptol:
            break
    t = delaunay_mesh(p, fd, geps)
    used = np.unique(t)
    remap = np.full(len(p), -1)
    remap[used] = np.arange(len(used))
    return p[used], remap[t]


def boundary(nodes, simplices):
    """
    Boundary of a tetrahedral (or triangular) mesh: the faces that belong
    to one simplex only, ordered so that their normals point out of the
    mesh. For tetrahedra these are the triangles that pytta.bem takes.
    """
    dim = simplices.shape[1] - 1
    faces, opposite = [], []
    for k in range(dim + 1):
        faces.append(np.delete(simplices, k, axis=1))
        opposite.append(simplices[:, k])
    faces, opposite = np.vstack(faces), np.concatenate(opposite)
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    once = counts[inverse] == 1
    faces, opposite = faces[once], opposite[once]
    P = nodes[faces]
    if dim == 3:
        normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    else:
        e = P[:, 1] - P[:, 0]
        normal = np.column_stack([e[:, 1], -e[:, 0]])
    flip = np.sum(normal * (nodes[opposite] - P[:, 0]), axis=1) > 0
    faces[flip] = faces[flip][:, ::-1]
    return faces


def read_mesh(filename, cell='tetra'):
    """
    Nodes and cells of one type ('tetra', 'triangle', ...) from a mesh file
    of Gmsh, COMSOL, Abaqus, VTK, Exodus and the other formats of meshio
    (an optional dependency, imported on call).
    """
    try:
        import meshio
    except ImportError as err:          # pragma: no cover
        raise ImportError('read_mesh needs the meshio package '
                          '(pip install meshio).') from err
    m = meshio.read(filename)
    cells = [c.data for c in m.cells if c.type == cell]
    if not cells:
        raise ValueError(f'The file has no {cell} cells.')
    return np.asarray(m.points, float), np.vstack(cells)


def quality(nodes, simplices):
    """Radius ratio of each simplex, dim r_in / r_out: 1 for the regular
    triangle or tetrahedron, 0 for a degenerate one. In 2D it is
    (b + c - a)(c + a - b)(a + b - c) / (abc) (Persson and Strang, eq.
    7.1), and all q > 0.5 is their rule of thumb for a good mesh."""
    P = nodes[simplices]
    dim = simplices.shape[1] - 1
    if dim == 2:
        a, b, c = (np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in ((1, 2), (2, 0), (0, 1)))
        return (b + c - a) * (c + a - b) * (a + b - c) / (a * b * c)
    V = np.abs(np.einsum('mi,mi->m', P[:, 1] - P[:, 0],
                         np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6
    faces = [P[:, [1, 2, 3]], P[:, [0, 2, 3]], P[:, [0, 1, 3]], P[:, [0, 1, 2]]]
    area = sum(np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1) / 2 for f in faces)
    r_in = 3 * V / area
    # Circumradius from the 3 x 3 system of the perpendicular bisectors.
    A = 2 * (P[:, 1:] - P[:, :1])
    b = np.sum(P[:, 1:] ** 2 - P[:, :1] ** 2, axis=2)
    centre = np.linalg.solve(A, b[..., None])[..., 0]
    r_out = np.linalg.norm(centre - P[:, 0], axis=1)
    return 3 * r_in / r_out
