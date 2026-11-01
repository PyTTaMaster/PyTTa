"""
Tests for pytta.mesh. The geometric tests pin each function on shapes with
known measures; the last two check the meshes by the eigenfrequencies and
the radiation of a sphere with pytta.fem (#122) and pytta.bem (#123), and
are skipped until those are present.
"""
import os
import tempfile
import unittest
import importlib.util

import numpy as np

import pytta
from pytta import mesh

HAS_FEM, HAS_BEM = hasattr(pytta, 'fem'), hasattr(pytta, 'bem')


def volume(p, t):
    P = p[t]
    if t.shape[1] == 3:
        return np.abs(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])).sum() / 2
    return np.abs(np.einsum('mi,mi->m', P[:, 1] - P[:, 0],
                            np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))).sum() / 6


def outward(p, faces, centre):
    P = p[faces]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    return np.all(np.sum(n * (P.mean(axis=1) - centre), axis=1) > 0)


class TestMesh(unittest.TestCase):

    def test_grid_points_warn_of_flat_simplices_inside(self):
        g = np.linspace(0, 1, 4)
        grid = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3)
        with self.assertWarns(UserWarning):
            mesh.delaunay_mesh(grid)

    def test_quality_of_regular_and_flat_simplices(self):
        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [2, 0]])
        tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [0, 0, 1.]])
        np.testing.assert_allclose(mesh.quality(tri, np.array([[0, 1, 2], [0, 1, 3]])), [1, 0], atol=1e-12)
        self.assertAlmostEqual(mesh.quality(tet, np.array([[0, 1, 2, 3]]))[0], 1, places=12)

    def test_delaunay_box_and_l_shape(self):
        # A 2 x 1 x 0.5 box filled from its corners and random points inside
        # and on its faces has its exact volume, and its boundary its exact
        # area with normals pointing out; an L-shape cut by its distance
        # function keeps its exact area.
        rng = np.random.default_rng(8)
        L = np.array([2, 1, 0.5])
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * L
        on_faces = rng.uniform(size=(60, 3)) * L
        on_faces[np.arange(60), rng.integers(0, 3, 60)] = 0
        pts = np.vstack([corners, rng.uniform(size=(80, 3)) * L, on_faces])
        t = mesh.delaunay_mesh(pts)
        s = mesh.boundary(pts, t)
        area = np.linalg.norm(np.cross(pts[s][:, 1] - pts[s][:, 0], pts[s][:, 2] - pts[s][:, 0]), axis=1).sum() / 2
        self.assertAlmostEqual(volume(pts, t), 1.0)
        self.assertAlmostEqual(area, 2 * (2 + 1 + 0.5))
        self.assertTrue(outward(pts, s, [1, 0.5, 0.25]))
        shape = mesh.difference(mesh.box([0, 0], [2, 2]), mesh.box([1, 1], [3, 3]))
        q = np.stack(np.meshgrid(np.linspace(0, 2, 9), np.linspace(0, 2, 9)), -1).reshape(-1, 2)
        self.assertAlmostEqual(volume(q, mesh.delaunay_mesh(q[shape(q) < 1e-9], shape)), 3.0)

    def test_distmesh_circle_meets_the_rule_of_thumb(self):
        # Every triangle above q = 0.5 (Persson and Strang), and the area
        # of the inscribed polygon approaching pi as h0^2.
        errors = []
        for h in (0.2, 0.1):
            p, t = mesh.distmesh(mesh.sphere([0, 0], 1.), h, [[-1, -1], [1, 1]])
            q = mesh.quality(p, t)
            errors.append(abs(volume(p, t) / np.pi - 1))
            self.assertGreater(q.min(), 0.5)
        print(f'\n  circle: area error {errors[0]:.2%} and {errors[1]:.2%}', end='')
        self.assertTrue(3 < errors[0] / errors[1] < 5)

    def test_distmesh_ball(self):
        # Volume and surface of the inscribed polyhedron within a few per
        # cent, the surface triangles pointing out and closed (every edge
        # shared by two of them).
        p, t = mesh.distmesh(mesh.sphere([0, 0, 0], 1.), 0.2, [[-1, -1, -1], [1, 1, 1]])
        s = mesh.boundary(p, t)
        edges = np.sort(np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        q = mesh.quality(p, t)
        print(f'\n  ball: volume {volume(p, t) / (4 / 3 * np.pi):.4f} of 4 pi / 3, '
              f'quality min {q.min():.3f}, mean {q.mean():.3f}', end='')
        self.assertLess(abs(volume(p, t) / (4 / 3 * np.pi) - 1), 0.04)
        self.assertTrue(outward(p, s, [0, 0, 0]))
        self.assertTrue(np.all(counts == 2))

    @unittest.skipUnless(importlib.util.find_spec('meshio'), 'read_mesh needs meshio')
    def test_read_mesh_round_trip(self):
        import meshio
        p, t = mesh.distmesh(mesh.sphere([0, 0, 0], 1.), 0.4, [[-1, -1, -1], [1, 1, 1]])
        with tempfile.TemporaryDirectory() as d:
            name = os.path.join(d, 'ball.vtu')
            meshio.write(name, meshio.Mesh(p, [('tetra', t)]))
            q, s = mesh.read_mesh(name)
        np.testing.assert_allclose(q, p)
        np.testing.assert_array_equal(s, t)

    @unittest.skipUnless(HAS_FEM, 'needs pytta.fem (#122)')
    def test_fem_modes_of_the_unit_ball(self):
        # Rigid unit ball: ka = 2.0816 (l = 1, three modes), 3.3421 (l = 2,
        # five) and 4.4934 (l = 0). The errors fall as h0^2, with the
        # slivers that the paper warns of left in the finer mesh.
        exact = np.r_[[2.0816] * 3, [3.3421] * 5, 4.4934]
        errors = []
        for h in (0.25, 0.125):
            p, t = mesh.distmesh(mesh.sphere([0, 0, 0], 1.), h, [[-1, -1, -1], [1, 1, 1]])
            ka = 2 * np.pi * pytta.fem.modes(p, t, 10, c=343.)[0][1:] / 343.
            errors.append(np.abs(ka / exact - 1))
        print(f'\n  ball modes: {np.round(100 * errors[0][[0, 3, 8]], 2)} % and '
              f'{np.round(100 * errors[1][[0, 3, 8]], 2)} %', end='')
        self.assertLess(errors[1].max(), 0.02)
        self.assertTrue(np.all(errors[0] / errors[1] > 3.5))

    @unittest.skipUnless(HAS_BEM, 'needs pytta.bem (#123)')
    def test_bem_on_the_surface_of_a_ball(self):
        # The pulsating sphere of radius 0.1 m at ka = 1 on the boundary of
        # the volume mesh: field within 0.5 % on the finer mesh.
        a, c, rho, k = 0.1, 343., 1.21, 10.
        points = np.array([[0, 0, 1.], [0.5, 0.2, 0.1]])
        r = np.linalg.norm(points, axis=1)
        exact = rho * c * 1j * k * a / (1 + 1j * k * a) * a / r * np.exp(-1j * k * (r - a))
        p, t = mesh.distmesh(mesh.sphere([0, 0, 0], 1.), 0.125, [[-1, -1, -1], [1, 1, 1]])
        nodes, tris = a * p, mesh.boundary(p, t)
        f = k * c / (2 * np.pi)
        pf = pytta.bem.field(nodes, tris, f, pytta.bem.surface_pressure(nodes, tris, f, 1.), 1., points)
        e = np.abs(pf / exact - 1).max()
        print(f'\n  BEM on the ball surface: {e:.1e}', end='')
        self.assertLess(e, 0.005)


if __name__ == '__main__':
    unittest.main()
