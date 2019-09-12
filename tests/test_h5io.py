import unittest
import pytta


class TestH5IO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        It runs before the bunch of tests
        """
        print('---> setUpClass')

    @classmethod
    def tearDownClass(cls):
        """
        It runs after the bunch of tests
        """
        print('\n---> tearDownClass')

    def setUp(self):
        """
        It runs first before each test
        """
        self.filename = 'h5teste.hdf5'

    def tearDown(self):
        """
        It runs after each test
        """
        pass

    def test_h5save_signalobj(self):
        """
        SignalObj hdf5 save test
        """
        sin1 = pytta.generate.sin(freq=500, timeLength=6)
        sin2 = pytta.generate.sin(freq=1000, timeLength=7)
        savedlst = [sin1, sin2]
        pytta.save(self.filename, sin1, sin2)
        loaded = pytta.load(self.filename)
        loadedlst = [loaded[pyttaobj] for pyttaobj in loaded]
        for idx, pobj in enumerate(loadedlst):
            # Testing every attribute
            # SignalObj.timeSignal
            self.assertSequenceEqual(pobj.timeSignal.tolist(),
                                     savedlst[idx].timeSignal.tolist())
            # SignalObj.samplingRate
            self.assertEqual(pobj.samplingRate,
                             savedlst[idx].samplingRate)
            # SignalObj.channels
            self.assertEqual(str(pobj.channels),
                             str(savedlst[idx].channels))

    def test_h5save_impulsiveresponse(self):
        """"
        ImpulsiveResponse hdf5 save test
        """
        xt = pytta.generate.sweep(fftDegree=16)
        noise = pytta.generate.noise(fftDegree=16, startMargin=0, stopMargin=0)
        noise *= 0.5
        yt = xt + noise
        IR = pytta.ImpulsiveResponse(excitationSignal=xt,
                                     recordedSignal=yt,
                                     method='Ht', winType='hann',
                                     winSize=44100*0.1, overlap=0.6,
                                     samplingRate=noise.samplingRate,
                                     freqMax=10000, freqMin=100,
                                     comment='testing the stuff')

        pytta.save(self.filename, IR)

        a = pytta.load(self.filename)
        loadedlst = [a[pyttaobj] for pyttaobj in a]
        self.assertSequenceEqual(loadedlst[0].systemSignal.timeSignal.tolist(),
                                 IR.systemSignal.timeSignal.tolist())

    def test_h5save_recmeasure(self):
        """
        RecMeasure hdf5 save test
        """
        # TO DO
        # # %% RecMeasure save test
        # med = pytta.generate.measurement('rec',
        #                                 freqMin=20,
        #                                 freqMax=20000,
        #                                 lengthDomain='samples',
        #                                 fftDegree=18)

        # self.filename = 'h5teste.hdf5'

        # pytta.h5save(self.filename, med)

        # a = pytta.h5load(self.filename)


def test_h5save_playrecmeasure(self):
        """
        PlayRecMeasure hdf5 save test
        """
        # TO DO
        # # %% PlayRecMeasure save test
        # sweep = pytta.generate.sweep()
        # med = pytta.generate.measurement('playrec',freqMin=20,freqMax=20000,
        # excitation=sweep)

        # self.filename = 'h5teste.hdf5'

        # pytta.h5save(self.filename, med)

        # a = pytta.h5load(self.filename)


def test_h5save_frfmeasure(self):
        """
        FRFMeasure hdf5 save test
        """
        # TO DO
        # # %% FRFMeasure save test
        # sweep = pytta.generate.sweep()
        # med = pytta.generate.measurement('frf',freqMin=20,freqMax=20000,
        # excitation=sweep)

        # self.filename = 'h5teste.hdf5'

        # pytta.h5save(self.filename, med)

        # a = pytta.h5load(self.filename)


if __name__ == '__main__':
    unittest.main()
