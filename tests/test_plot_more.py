"""
Tests for plot_group_delay, plot_nyquist, plot_zplane and plot_polar of
pytta.functions, which follow ita_plot_groupdelay, ita_plot_nyquist,
ita_plot_zplane and ita_plot_polar of the ITA-Toolbox. Each test reads back
the plotted data and compares it with an analytic result.
"""
import os
import unittest

import numpy as np
import scipy.signal as ss

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import pytta

FS = 8000


def delayed_impulse(d, n=1024):
    label = 'x'
    x = np.zeros(n)
    x[d] = 1
    return pytta.SignalObj(x, 'time', FS)


class TestPlotMore(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_group_delay_of_a_pure_delay(self):
        # A delay of 40 samples is 5 ms at every frequency.
        y = pytta.plot_group_delay(delayed_impulse(40)).axes[0].lines[0].get_ydata()
        print(f'\n  group delay: {np.abs(y - 5).max():.1e} ms off 5 ms', end='')
        self.assertLess(np.abs(y - 5).max(), 1e-9)

    def test_nyquist_of_a_pure_delay_is_a_circle(self):
        sig = delayed_impulse(7)
        line = pytta.plot_nyquist(sig).axes[0].lines[0]
        # The spectrum of SignalObj scales the bins above DC alike, so they
        # lie on one circle.
        r = np.hypot(line.get_xdata(), line.get_ydata())[1:]
        print(f'\n  nyquist: radius spread {np.ptp(r) / r.mean():.1e}', end='')
        self.assertLess(np.ptp(r) / r.mean(), 1e-6)

    def test_zplane_marks_the_poles_and_zeros(self):
        z, p, _ = ss.tf2zpk(*ss.butter(4, 0.2))
        lines = pytta.plot_zplane(z, p).axes[0].lines
        np.testing.assert_allclose(lines[1].get_xdata() + 1j * lines[1].get_ydata(), z)
        np.testing.assert_allclose(lines[2].get_xdata() + 1j * lines[2].get_ydata(), p)

    def test_polar_curve_closes_at_the_front(self):
        angles = np.arange(0, 360, 30)
        levels = np.column_stack([np.cos(np.deg2rad(angles)), np.ones(12)])
        ax = pytta.plot_polar(angles, levels, labels=['a', 'b']).axes[0]
        theta, r = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        self.assertEqual(ax.get_theta_offset(), np.pi / 2)
        self.assertEqual((theta.size, r[-1]), (13, r[0]))


if __name__ == '__main__':
    unittest.main()
