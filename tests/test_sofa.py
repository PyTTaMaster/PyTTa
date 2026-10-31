"""Tests for pytta.sofa (AES69-2022, SOFA 2.1)."""

import os
import tempfile
import unittest

import h5py
import numpy as np
from scipy.io import loadmat

import pytta
from pytta import sofa

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')

# Items flagged 'm' or 'rm' in SimpleFreeFieldHRIR_1.0.csv of the SOFA
# conventions (sofaconventions.org, SOFAtoolbox/conventions).
HRIR_GLOBALS = ['Conventions', 'Version', 'SOFAConventions',
                'SOFAConventionsVersion', 'APIName', 'APIVersion',
                'AuthorContact', 'DataType', 'License', 'Organization',
                'RoomType', 'DateCreated', 'DateModified', 'Title',
                'DatabaseName', 'ListenerShortName']
HRIR_VARIABLES = {'ListenerPosition': ('I', 'C'),
                  'ReceiverPosition': ('R', 'C', 'I'),
                  'SourcePosition': ('M', 'C'),
                  'EmitterPosition': ('E', 'C', 'I'),
                  'ListenerUp': ('I', 'C'),
                  'ListenerView': ('I', 'C'),
                  'Data.IR': ('M', 'R', 'N'),
                  'Data.SamplingRate': ('I',),
                  'Data.Delay': ('I', 'R')}
HRIR_VARIABLE_ATTRS = ['ListenerPosition:Type', 'ListenerPosition:Units',
                       'ReceiverPosition:Type', 'ReceiverPosition:Units',
                       'SourcePosition:Type', 'SourcePosition:Units',
                       'EmitterPosition:Type', 'EmitterPosition:Units',
                       'ListenerView:Type', 'ListenerView:Units',
                       'Data.SamplingRate:Units']


class TestSofa(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_keeps_data_fs_and_positions(self):
        """GeneralFIR with 4 receivers: write then read is exact."""
        rng = np.random.default_rng(0)
        ir = rng.standard_normal((3, 4, 64)).astype('float32')
        signals = []
        for m in range(3):
            signals.append(pytta.SignalObj(signalArray=ir[m].T,
                                           domain='time',
                                           samplingRate=32000))
        src = np.array([[1.0, 0.5, 0.0], [0.0, 2.25, 1.5], [-1.0, 0.0, 0.3]])
        path = os.path.join(self.tmp.name, 'rt.sofa')
        sofa.write(path, signals, src, convention='GeneralFIR',
                   Title='round trip')
        out = sofa.read(path)
        d_ir = np.max(np.abs(out['ir'] - ir))
        d_sig = max(np.max(np.abs(s.timeSignal - ir[m].T))
                    for m, s in enumerate(out['signal']))
        d_src = np.max(np.abs(out['source_position']['values'] - src))
        print(f'\nround trip: max |dIR| = {d_ir}, max |dSignal| = {d_sig}, '
              f'max |dSourcePosition| = {d_src}, fs = {out["fs"]}')
        self.assertEqual(d_ir, 0.0)
        self.assertEqual(d_sig, 0.0)
        self.assertEqual(d_src, 0.0)
        self.assertEqual(out['fs'], 32000)
        self.assertEqual(out['source_position']['Type'], 'cartesian')
        self.assertEqual(out['attributes']['Title'], 'round trip')
        self.assertEqual(out['attributes']['SOFAConventions'], 'GeneralFIR')

    def test_hrir_has_every_mandatory_item(self):
        """The written file has all 'm'/'rm' items of SimpleFreeFieldHRIR."""
        signals = []
        for az in (0, 90):
            signals.append(pytta.SignalObj(signalArray=np.ones((16, 2)),
                                           domain='time',
                                           samplingRate=48000))
        path = os.path.join(self.tmp.name, 'hrir.sofa')
        sofa.write(path, signals, [[0, 0, 1.2], [90, 0, 1.2]])
        sizes = {'M': 2, 'R': 2, 'N': 16, 'C': 3, 'I': 1, 'E': 1}
        with h5py.File(path, 'r') as f:
            missing = [a for a in HRIR_GLOBALS if a not in f.attrs]
            missing += [v for v in HRIR_VARIABLES if v not in f]
            missing += [a for a in HRIR_VARIABLE_ATTRS
                        if a.split(':')[0] not in f
                        or a.split(':')[1] not in f[a.split(':')[0]].attrs]
            wrong = [v for v, dims in HRIR_VARIABLES.items() if v in f
                     and f[v].shape != tuple(sizes[d] for d in dims)]
            convention = f.attrs['SOFAConventions']
        print(f'\nmissing mandatory items: {missing}, wrong shapes: {wrong}')
        self.assertEqual(missing, [])
        self.assertEqual(wrong, [])
        self.assertEqual(convention, b'SimpleFreeFieldHRIR')

    def test_read_matches_matlab_ncread(self):
        """A SOFA file written by MATLAB reads as MATLAB ncread reads it."""
        ref = loadmat(os.path.join(GOLDEN, 'sofa_golden.mat'))
        out = sofa.read(os.path.join(GOLDEN, 'sofa_golden.sofa'))
        d_ir = np.max(np.abs(out['ir'] - ref['IR']))
        d_src = np.max(np.abs(out['source_position']['values']
                              - ref['SourcePosition']))
        d_rec = np.max(np.abs(out['receiver_position']['values']
                              - ref['ReceiverPosition'].reshape(2, 3, 1)))
        d_sig = max(np.max(np.abs(s.timeSignal - ref['IR'][m].T))
                    for m, s in enumerate(out['signal']))
        print(f'\nvs ncread: max |dIR| = {d_ir}, max |dSource| = {d_src}, '
              f'max |dReceiver| = {d_rec}, max |dSignal| (float32) = {d_sig}')
        self.assertEqual(d_ir, 0.0)
        self.assertEqual(d_src, 0.0)
        self.assertEqual(d_rec, 0.0)
        self.assertLess(d_sig, 1e-7)
        self.assertEqual(out['fs'], ref['SamplingRate'].item())
        np.testing.assert_array_equal(out['delay'], ref['Delay'])
        self.assertEqual(out['source_position']['Type'],
                         str(ref['SourcePositionType'][0]))
        self.assertEqual(out['source_position']['Units'],
                         str(ref['SourcePositionUnits'][0]))
        self.assertEqual(out['attributes']['Title'], str(ref['Title'][0]))


if __name__ == '__main__':
    unittest.main()
