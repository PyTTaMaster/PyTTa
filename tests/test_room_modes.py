"""
Tests for the modal model of a rectangular room in pytta.rooms
(rectangular_room_frf, rectangular_room_ir), after
ita_roomacoustics_analytic_FRF of the ITA-Toolbox. Each test pins one rule
of the modal superposition (Fahy, Foundations of Engineering Acoustics,
9.8).
"""
import unittest

import numpy as np

from pytta import rooms

L = (5., 4., 3.)
C = 343.2


class TestRoomModes(unittest.TestCase):

    def test_isolated_mode_frequency_and_bandwidth(self):
        # The (1, 0, 0) mode sits at c / (2 Lx) = 34.32 Hz, alone up to the
        # (0, 1, 0) mode at 42.9 Hz; its half-power bandwidth is
        # d / pi = 3 ln(10) / (pi T) = 2.2 / T.
        T = 2.
        f = np.arange(25, 39, 0.001)
        H = np.abs(rooms.rectangular_room_frf(L, (0, 2, 1.5), (5, 2, 1.5), f, T, f_max=200))
        peak = f[np.argmax(H)]
        band = f[H >= H.max() / np.sqrt(2)]
        width = band[-1] - band[0]
        print(f'\n  peak {peak:.3f} Hz (mode {C / 10:.3f} Hz), half-power width {width:.4f} Hz '
              f'(3 ln 10 / (pi T) = {3 * np.log(10) / (np.pi * T):.4f} Hz)')
        self.assertAlmostEqual(peak, C / 10, delta=0.05)
        self.assertAlmostEqual(width, 3 * np.log(10) / (np.pi * T), delta=0.05)

    def test_reciprocity_and_nodal_plane(self):
        # Swapping source and receiver leaves H unchanged; a receiver on
        # the plane x = Lx / 2 does not excite the (1, 0, 0) mode.
        f = np.linspace(10, 120, 500)
        a = rooms.rectangular_room_frf(L, (0.7, 1.1, 0.4), (3.9, 2.7, 2.2), f)
        b = rooms.rectangular_room_frf(L, (3.9, 2.7, 2.2), (0.7, 1.1, 0.4), f)
        fm = np.array([C / 10])
        at_node = abs(rooms.rectangular_room_frf(L, (0, 2, 1.5), (2.5, 2, 1.5), fm, f_max=60)[0])
        off_node = abs(rooms.rectangular_room_frf(L, (0, 2, 1.5), (5, 2, 1.5), fm, f_max=60)[0])
        print(f'\n  reciprocity {np.abs(a - b).max():.1e}; at the mode, receiver on the nodal plane '
              f'{20 * np.log10(at_node / off_node):.1f} dB below the wall')
        self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-12)
        self.assertLess(at_node / off_node, 0.1)

    def test_impulse_response_decays_with_T(self):
        # The modal damping d = 3 ln(10) / T makes the energy fall 60 dB in
        # T. The decay curve of the energy summed over six receivers (one
        # receiver beats between modes) gives T back from -5 to -35 dB, and
        # it keeps falling with no floor: from 2.0 s to 2.5 s it lies more
        # than 100 dB below its first 0.25 s.
        label = 'decay'
        T, fs, n = 0.8, 2000, 6000
        rng = np.random.default_rng(2)
        e = 0
        for k in range(6):
            r = rng.uniform((0.3, 0.3, 0.3), (4.7, 3.7, 2.7))
            ir = rooms.rectangular_room_ir(L, (0.7, 1.1, 0.4), r, fs, n, T)
            e = e + ir.timeSignal[:, 0].astype(float) ** 2
        edc = 10 * np.log10(np.cumsum(e[::-1])[::-1] / e.sum())
        t = np.arange(n) / fs
        T30 = 2 * (t[np.argmax(edc < -35)] - t[np.argmax(edc < -5)])
        tail = 10 * np.log10(e[4000:5000].mean() / e[:500].mean())
        print(f'\n  T from the decay {T30:.4f} s, T of the model {T} s; 2.0 to 2.5 s at {tail:.1f} dB')
        self.assertAlmostEqual(T30 / T, 1, delta=0.05)
        self.assertLess(tail, -100)

    def test_rejects_positions_outside(self):
        with self.assertRaises(ValueError):
            rooms.rectangular_room_frf(L, (6, 1, 1), (1, 1, 1), [100.])


if __name__ == '__main__':
    unittest.main()
