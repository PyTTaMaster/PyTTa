"""
Tests for pytta.rooms.image_source_ir, the image source method of Allen and
Berkley (1979) for a rectangular room, after ita_roomacoustics_imagesources
of the ITA-Toolbox. Each test pins one rule.
"""
import unittest

import numpy as np

from pytta import rooms

C, FS = 343.2, 16000


def spectrum(ir):
    h = ir.timeSignal[:, 0].astype(float)
    return np.fft.rfftfreq(h.size, 1 / FS), np.fft.rfft(h)


class TestImageSources(unittest.TestCase):

    def test_anechoic_and_one_wall(self):
        # With no reflection the response is the free-field Green's
        # function exp(-j w d / c) / (4 pi d); with only the wall x = 0
        # reflecting (factor 1) the mirror image at (-x0, y0, z0) adds its
        # own spherical wave. Compared below FS / 4, where the windowed sinc
        # that places each arrival is flat.
        label = 'image'
        Lr, s, r = (6., 5., 4.), (1.2, 2.0, 1.5), (4.1, 3.2, 2.6)
        d0 = np.linalg.norm(np.subtract(s, r))
        d1 = np.linalg.norm(np.subtract((-s[0], s[1], s[2]), r))
        free = rooms.image_source_ir(Lr, s, r, FS, 0.05, reflection=0)
        wall = rooms.image_source_ir(Lr, s, r, FS, 0.05, reflection=[1, 0, 0, 0, 0, 0])
        f, Hf = spectrum(free)
        _, Hw = spectrum(wall)
        band = (f > 50) & (f < FS / 4)
        w = 2 * np.pi * f[band]
        g0 = np.exp(-1j * w * d0 / C) / (4 * np.pi * d0)
        g1 = np.exp(-1j * w * d1 / C) / (4 * np.pi * d1)
        e_free = np.abs(Hf[band] - g0).max() / np.abs(g0).max()
        e_wall = np.abs(Hw[band] - g0 - g1).max() / np.abs(g0).max()
        print(f'\n  anechoic {e_free:.1e}, one wall {e_wall:.1e} relative to 1 / (4 pi d)')
        self.assertLess(e_free, 2e-3)
        self.assertLess(e_wall, 2e-3)

    def test_each_wall_weights_its_images(self):
        # Walls x = 0 and y = 0 reflecting with 0.5 and 0.8 and the others
        # absorbing: the response is the direct sound, the images across
        # each wall (weights 0.5 and 0.8) and the corner image (0.5 x 0.8).
        # The wall x = Lx alone puts the image at 2 Lx - x0.
        label = 'walls'
        Lr, s, r = (6., 5., 4.), (1.2, 2.0, 1.5), (4.1, 3.2, 2.6)
        f, H = spectrum(rooms.image_source_ir(Lr, s, r, FS, 0.05, [0.5, 0, 0.8, 0, 0, 0]))
        _, Hx = spectrum(rooms.image_source_ir(Lr, s, r, FS, 0.05, [0, 1, 0, 0, 0, 0]))
        band = (f > 50) & (f < FS / 4)
        w = 2 * np.pi * f[band]

        def wave(pos, weight):
            d = np.linalg.norm(np.subtract(pos, r))
            return weight * np.exp(-1j * w * d / C) / (4 * np.pi * d)
        x0, y0, z0 = s
        expected = (wave(s, 1) + wave((-x0, y0, z0), 0.5) + wave((x0, -y0, z0), 0.8)
                    + wave((-x0, -y0, z0), 0.4))
        expected_x = wave(s, 1) + wave((2 * Lr[0] - x0, y0, z0), 1)
        ref = np.abs(wave(s, 1)).max()
        e = np.abs(H[band] - expected).max() / ref
        ex = np.abs(Hx[band] - expected_x).max() / ref
        print(f'\n  two walls with their corner {e:.1e}, wall x = Lx {ex:.1e}')
        self.assertLess(e, 2e-3)
        self.assertLess(ex, 2e-3)

    def test_rejects_positions_outside(self):
        with self.assertRaises(ValueError):
            rooms.image_source_ir((5, 4, 3), (5.5, 1, 1), (1, 1, 1), FS, 0.1)


if __name__ == '__main__':
    unittest.main()
