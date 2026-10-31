"""
Tests for pytta.iso532_2, the Moore-Glasberg loudness of ISO 532-2:2017.
Each test pins one rule.

The oracle is the standard itself: the results of Annex B, which the program
of Annex C produced, and Table 5, which 8.2 defines as the loudness of a
1 kHz tone in free field heard with both ears. Both are printed with two to
four significant digits, so a value passes when it lies within one unit of
the last printed digit.
"""
import unittest

import numpy as np

from pytta import iso532_2 as M


def close(tc, got, ref, label):
    """Within one unit of the last digit of ref (given as a string)."""
    unit = 10.0 ** -(len(ref.split('.')[1]) if '.' in ref else 0)
    print(f'\n  {label}: {got:.4f} against {ref}', end='')
    tc.assertLessEqual(abs(got - float(ref)), unit + 1e-12, label)


class TestISO532_2(unittest.TestCase):

    def test_annex_b_tones(self):
        # B.1.1 to B.1.4 and B.3. B.1.1 prints 8.1 sone at 70 dB, while
        # Table 5 of the same standard gives 8.04 for 70 phon; 8.04 is
        # within one unit of 8.1.
        for L, ref in zip(range(10, 90, 10), ('0.03', '0.14', '0.43', '1.0', '2.1', '4.1', '8.1', '15.8')):
            close(self, M.loudness([1000.], [L])['N'], ref, f'B.1.1 1 kHz {L} dB')
        for L, ref, ref_LN in ((20, '0.35', '28'), (40, '1.8', '48'), (60, '7.0', '68'), (80, '27.2', '87.5')):
            out = M.loudness([3000.], [L])
            close(self, out['N'], ref, f'B.1.2 3 kHz {L} dB')
            close(self, out['LN'], ref_LN, f'B.1.2 3 kHz {L} dB, phon')
        for L, ref in zip((20, 40, 60, 80), ('0.07', '0.54', '2.31', '8.82')):
            close(self, M.loudness([1000.], [L], 'eardrum', 'one')['N'], ref, f'B.1.3 earphone {L} dB')
        close(self, M.loudness([100.], [50])['N'], '0.351', 'B.1.4 100 Hz 50 dB')
        close(self, M.loudness([1500., 1600, 1700], [60] * 3)['N'], '6.31', 'B.3.1')
        close(self, M.loudness([1000., 1600, 2400], [60] * 3)['N'], '12.49', 'B.3.2')
        close(self, M.loudness(np.arange(100., 1001, 100), [30] * 10)['N'], '2.00', 'B.3.3')

    def test_annex_b_noise(self):
        # B.2.1 to B.2.5: bands of white and pink noise (5.3) and flat
        # one-third octave levels (5.5, the scheme of the program). B.2.5 at
        # 0 dB is left out: the program reads a band level of 0 as silence,
        # so that printed value (0.0004 sone) cannot come from it.
        for f1, f2, s, ref in ((950, 1050, 40, '4.21'), (500, 1500, 40, '14.17'),
                               (500, 1500, 30, '7.97')):
            close(self, M.loudness(*M.noise_band_components(f1, f2, s))['N'], ref,
                  f'B.2.1/2 white {f1}-{f2} Hz')
        for s, ref in ((0, '3.64'), (20, '15.85'), (40, '48.59')):
            close(self, M.loudness(*M.noise_band_components(50, 15000, s, pink=True))['N'],
                  ref, f'B.2.3 pink {s} dB at 1 kHz')
        for L, ref in zip((0, 10, 20, 30, 40, 50), ('0.077', '0.69', '2.54', '6.25', '12.6', '23.1')):
            close(self, M.loudness(*M.third_octave_components(np.full(29, L)))['N'], ref,
                  f'B.2.4 bands at {L} dB')
        for L, ref in zip((10, 20, 30, 40, 50), ('0.08', '0.72', '2.41', '5.55', '10.7')):
            close(self, M.loudness(*M.third_octave_components(np.full(29, L)), 'eardrum', 'one')['N'],
                  ref, f'B.2.5 bands at {L} dB')

    def test_table5_is_the_loudness_of_a_1_khz_tone(self):
        # 8.2: a 1 kHz tone, free field, both ears, at L dB has the
        # loudness of L phon in Table 5; loudness_level inverts the table.
        printed = ('0.001 0.004 0.008 0.010 0.019 0.031 0.073 0.146 0.26 0.43 0.67 1.00 '
                   '1.46 2.09 2.96 4.14 5.77 8.04 11.2 15.8 22.7 32.9 47.7 69.6 102.0 151.0 '
                   '225.0 337.6').split()
        for (LN, _), ref in zip(M._TABLE5, printed):
            close(self, M.loudness([1000.], [LN])['N'], ref, f'Table 5, {LN} phon')
        np.testing.assert_allclose(M.loudness_level(M._TABLE5[:, 1]), M._TABLE5[:, 0], atol=1e-12)

    def test_binaural_inhibition(self):
        # 8.1: the same sound at both ears is 1.5 times as loud as at one
        # ear (theta makes sech(1)^theta = 1/2); a sound at one ear only is
        # not inhibited, so its binaural loudness is the monaural one.
        f, L = M.noise_band_components(500, 1500, 40)
        mono = M.loudness(f, L, ears='one')['N']
        diotic = M.loudness(f, L)['N']
        one_ear = M.loudness(f, np.column_stack([L, np.full(L.size, -np.inf)]))['N']
        print(f'\n  diotic / monaural {diotic / mono:.6f}, one ear / monaural {one_ear / mono:.9f}')
        self.assertAlmostEqual(diotic / mono, 1.5, delta=1e-4)
        self.assertAlmostEqual(one_ear / mono, 1, delta=1e-9)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            M.loudness([1000.], [60], field='near')
        with self.assertRaises(ValueError):
            M.loudness([1000., 2000.], [60])
        with self.assertRaises(ValueError):
            M.loudness([1000.], [[60, 60]], ears='one')
        with self.assertRaises(ValueError):
            M.third_octave_components(np.zeros(28))


if __name__ == '__main__':
    unittest.main()
