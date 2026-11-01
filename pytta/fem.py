# -*- coding: utf-8 -*-
"""
Finite element method for the Helmholtz equation in air, with linear
tetrahedra, as the fluid FEM of the ITA-Toolbox:

    (K - k^2 M + j k sum_s beta_s Ms) p = j w rho sum_s Ns u_s

with K and M the stiffness and mass matrices of the fluid, Ms the mass
matrix of a boundary surface of normalized admittance beta_s = rho c / Zs
and Ns u_s the load of a surface vibrating with normal velocity u_s into
the fluid (time dependence exp(j w t)). The matrices follow Zienkiewicz,
Taylor and Zhu, The Finite Element Method, and the formulation Ihlenburg,
Finite Element Analysis of Acoustic Scattering (1998).

>>> nodes, tets, faces = pytta.fem.box_mesh(5, 4, 3, 25, 20, 15)
>>> f, shapes = pytta.fem.modes(nodes, tets, 10)
>>> p = pytta.fem.harmonic(nodes, tets, 100., velocity={'x0': 1e-3},
...                        admittance={'x1': 1.}, faces=faces)

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ['box_mesh', 'assemble', 'surface_mass', 'modes', 'harmonic']

# The six tetrahedra of a hexahedron, by the local numbering of its corners
# (bit 0 x, bit 1 y, bit 2 z), all sharing the diagonal 0-7.
_HEX_TETS = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                      [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])


def box_mesh(Lx, Ly, Lz, nx, ny, nz):
    """
    Tetrahedral mesh of the box [0, Lx] x [0, Ly] x [0, Lz] with nx, ny, nz
    hexahedra along each side, each split into six tetrahedra. Returns
    (nodes (N, 3), tetrahedra (M, 4), faces), faces being a dict of the
    boundary triangles (F, 3) of each side: 'x0', 'x1', 'y0', 'y1', 'z0'
    and 'z1'.
    """
    x, y, z = (np.linspace(0, L, n + 1) for L, n in ((Lx, nx), (Ly, ny), (Lz, nz)))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nodes.shape[0]).reshape(nx + 1, ny + 1, nz + 1)
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    corners = np.stack([idx[i + (c & 1), j + (c >> 1 & 1), k + (c >> 2 & 1)].ravel()
                        for c in range(8)], axis=1)
    tets = corners[:, _HEX_TETS].reshape(-1, 4)
    faces = {}
    for name, axis, value in (('x0', 0, 0), ('x1', 0, Lx), ('y0', 1, 0),
                              ('y1', 1, Ly), ('z0', 2, 0), ('z1', 2, Lz)):
        on = np.isclose(nodes[:, axis], value)
        tris = []
        for t in tets:
            free = t[on[t]]
            if free.size == 3:
                tris.append(free)
        faces[name] = np.array(tris)
    return nodes, tets, faces


def _gradients(nodes, tets):
    # Volumes and gradients of the four linear shape functions.
    P = nodes[tets]                                    # (M, 4, 3)
    J = P[:, 1:] - P[:, :1]                            # (M, 3, 3)
    det = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    ref = np.array([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    G = np.einsum('mab,kb->mka', Jinv, ref)         # J^-1 grad_ref N
    return np.abs(det) / 6, G


def assemble(nodes, tets):
    """Stiffness K = int grad Ni . grad Nj and mass M = int Ni Nj of the
    fluid, as sparse matrices (N, N)."""
    V, G = _gradients(nodes, tets)
    Ke = V[:, None, None] * np.einsum('mib,mjb->mij', G, G)
    Me = V[:, None, None] / 20 * (np.ones((4, 4)) + np.eye(4))
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    n = nodes.shape[0]
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    M = sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(n, n))
    return K, M


def surface_mass(nodes, triangles):
    """Boundary mass matrix int Ni Nj dS of the triangles, sparse (N, N)."""
    P = nodes[triangles]
    area = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) / 2
    Me = area[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3))
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    n = nodes.shape[0]
    return sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(n, n))


def modes(nodes, tets, num=10, c=343.):
    """Eigenfrequencies (Hz) and mode shapes (N, num) of the rigid-walled
    cavity, from K phi = k^2 M phi."""
    K, M = assemble(nodes, tets)
    vals, vecs = spla.eigsh(K.tocsc(), num, M.tocsc(), sigma=-1.)
    order = np.argsort(vals)
    k = np.sqrt(np.maximum(vals[order], 0))
    return k * c / (2 * np.pi), vecs[:, order]


def harmonic(nodes, tets, freq, velocity=None, admittance=None, faces=None,
             c=343., rho=1.21):
    """
    Sound pressure at the nodes at freq (Hz). velocity maps surfaces (keys
    of faces, or triangle arrays) to the normal velocity (m/s) with which
    they move into the fluid, admittance maps surfaces to the normalized
    admittance rho c / Z; the other surfaces are rigid.
    """
    def tri(s):
        return faces[s] if isinstance(s, str) else np.asarray(s)

    K, M = assemble(nodes, tets)
    w = 2 * np.pi * freq
    k = w / c
    A = (K - k**2 * M).astype(complex)
    for s, beta in (admittance or {}).items():
        A = A + 1j * k * beta * surface_mass(nodes, tri(s))
    f = np.zeros(nodes.shape[0], dtype=complex)
    for s, u in (velocity or {}).items():
        f += 1j * w * rho * u * surface_mass(nodes, tri(s)).sum(axis=1).A1
    return spla.spsolve(A.tocsc(), f)
