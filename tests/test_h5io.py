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
        sin2.signalType = 'energy'
        savedlst = [sin1, sin2]
        pytta.save(self.filename, sin1, sin2)
        loaded = pytta.load(self.filename)
        loadedlst = [loaded[pyttaobj] for pyttaobj in loaded]
        for idx, pobj in enumerate(loadedlst):
            # Testing every attribute
            # SignalObj.timeSignal
            self.assertSequenceEqual(pobj.timeSignal.tolist(),
                                     savedlst[idx].timeSignal.tolist())
            # SignalObj.freqSignal
            self.assertSequenceEqual(pobj.freqSignal.tolist(),
                                     savedlst[idx].freqSignal.tolist())
            # SignalObj.samplingRate
            self.assertEqual(pobj.samplingRate,
                             savedlst[idx].samplingRate)
            # SignalObj.signalType
            self.assertEqual(pobj.signalType,
                             savedlst[idx].signalType)
            # SignalObj.channels
            self.assertEqual(str(pobj.channels),
                             str(savedlst[idx].channels))


    def test_h5save_impulsiveresponse(self):
        """"
        ImpulsiveResponse hdf5 save test
        """
        xt = pytta.generate.sweep(fftDegree=16)
        noise = pytta.generate.random_noise(fftDegree=16, startMargin=0, stopMargin=0)
        noise *= 0.5
        yt = xt + noise
        IR = pytta.ImpulsiveResponse(excitation=xt,
                                     recording=yt,
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
        mType = 'rec'
        lengthDomain = 'samples'
        lengthDomain2 = 'time'
        fftDegree = 18
        timeLength2 = 4
        device = 4
        device2 = 3
        inChannels = [1,2,3,4]
        inChannels2 = [1,5,3,7]
        samplingRate = 44100
        samplingRate2 = 48000
        freqMin = 22
        freqMin2 = 20
        freqMax = 19222
        freqMax2 = 20000
        comment = 'Testing'
        comment2 = 'Testing2'
        
        med1 = pytta.generate.measurement(kind=mType,
                                         samplingRate=samplingRate,
                                         freqMin=freqMin,
                                         freqMax=freqMax,
                                         device=device,
                                         inChannels=inChannels,
                                         comment=comment,
                                         lengthDomain=lengthDomain,
                                         fftDegree=fftDegree)

        med2 = pytta.generate.measurement(kind=mType,
                                         samplingRate=samplingRate2,
                                         freqMin=freqMin2,
                                         freqMax=freqMax2,
                                         device=device2,
                                         inChannels=inChannels2,
                                         comment=comment2,
                                         lengthDomain=lengthDomain2,
                                         timeLength=timeLength2)

        savedlst = [med1, med2]

        pytta.save(self.filename, med1, med2)

        a = pytta.load(self.filename)

        loadedlst = [a[pyttaobj] for pyttaobj in a]

        for idx, pobj in enumerate(loadedlst):
            self.assertEqual(pobj.lengthDomain,
                             savedlst[idx].lengthDomain)

            self.assertEqual(pobj.timeLength,
                             savedlst[idx].timeLength)

            self.assertEqual(pobj.fftDegree,
                             savedlst[idx].fftDegree)

            self.assertEqual(pobj.device,
                             savedlst[idx].device)
            
            self.assertEqual(str(pobj.inChannels),
                             str(savedlst[idx].inChannels))
            
            self.assertEqual(pobj.samplingRate,
                             savedlst[idx].samplingRate)

            self.assertEqual(pobj.freqMin,
                             savedlst[idx].freqMin)

            self.assertEqual(pobj.freqMax,
                             savedlst[idx].freqMax)
            
            self.assertEqual(pobj.comment,
                             savedlst[idx].comment)


    def test_h5save_playrecmeasure(self):
            """
            PlayRecMeasure hdf5 save test
            """
            mType = 'playrec'
            fftDegree = 18
            fftDegree2 = 19
            device = 4
            device2 = 3
            inChannels = [1,2,3,4]
            inChannels2 = [1,5,3,7]
            outChannels = [1,2]
            outChannels2 = [1,3]
            samplingRate = 44100
            samplingRate2 = 48000
            freqMin = 22
            freqMin2 = 20
            freqMax = 19222
            freqMax2 = 20000
            comment = 'Testing'
            comment2 = 'Testing2'
            excitation = pytta.generate.sweep(freqMin=freqMin,
                                            freqMax=freqMax,
                                            samplingRate=samplingRate,
                                            fftDegree=fftDegree)

            excitation2 = pytta.generate.sweep(freqMin=freqMin2,
                                            freqMax=freqMax2,
                                            samplingRate=samplingRate2,
                                            fftDegree=fftDegree2)
            
            med1 = pytta.generate.measurement(kind=mType,
                                            excitation=excitation,
                                            samplingRate=samplingRate,
                                            freqMin=freqMin,
                                            freqMax=freqMax,
                                            device=device,
                                            inChannels=inChannels,
                                            outChannels=outChannels,
                                            comment=comment)

            med2 = pytta.generate.measurement(kind=mType,
                                            excitation=excitation2,
                                            samplingRate=samplingRate2,
                                            freqMin=freqMin2,
                                            freqMax=freqMax2,
                                            device=device2,
                                            inChannels=inChannels2,
                                            outChannels=outChannels2,
                                            comment=comment2) 

            savedlst = [med1, med2]

            pytta.save(self.filename, med1, med2)

            a = pytta.load(self.filename)

            loadedlst = [a[pyttaobj] for pyttaobj in a]

            for idx, pobj in enumerate(loadedlst):
                self.assertEqual(pobj.lengthDomain,
                                savedlst[idx].lengthDomain)

                self.assertEqual(pobj.excitation.timeSignal.tolist(),
                                savedlst[idx].excitation.timeSignal.tolist())

                self.assertEqual(pobj.device,
                                savedlst[idx].device)
                
                self.assertEqual(str(pobj.inChannels),
                                str(savedlst[idx].inChannels))

                self.assertEqual(str(pobj.outChannels),
                                str(savedlst[idx].outChannels))
                
                self.assertEqual(pobj.samplingRate,
                                savedlst[idx].samplingRate)

                self.assertEqual(pobj.freqMin,
                                savedlst[idx].freqMin)

                self.assertEqual(pobj.freqMax,
                                savedlst[idx].freqMax)
                
                self.assertEqual(pobj.comment,
                                savedlst[idx].comment)


    def test_h5save_frfmeasure(self):
            """
            FRFMeasure hdf5 save test
            """
            mType = 'frf'
            fftDegree = 18
            fftDegree2 = 19
            device = 4
            device2 = 3
            inChannels = [1,2,3,4]
            inChannels2 = [1,5,3,7]
            outChannels = [1,2]
            outChannels2 = [1,3]
            samplingRate = 44100
            samplingRate2 = 48000
            freqMin = 22
            freqMin2 = 20
            freqMax = 19222
            freqMax2 = 20000
            comment = 'Testing'
            comment2 = 'Testing2'
            excitation = pytta.generate.sweep(freqMin=freqMin,
                                            freqMax=freqMax,
                                            samplingRate=samplingRate,
                                            fftDegree=fftDegree)

            excitation2 = pytta.generate.sweep(freqMin=freqMin2,
                                            freqMax=freqMax2,
                                            samplingRate=samplingRate2,
                                            fftDegree=fftDegree2)
            
            med1 = pytta.generate.measurement(kind=mType,
                                            excitation=excitation,
                                            samplingRate=samplingRate,
                                            freqMin=freqMin,
                                            freqMax=freqMax,
                                            device=device,
                                            inChannels=inChannels,
                                            outChannels=outChannels,
                                            comment=comment)

            med2 = pytta.generate.measurement(kind=mType,
                                            excitation=excitation2,
                                            samplingRate=samplingRate2,
                                            freqMin=freqMin2,
                                            freqMax=freqMax2,
                                            device=device2,
                                            inChannels=inChannels2,
                                            outChannels=outChannels2,
                                            comment=comment2) 

            savedlst = [med1, med2]

            pytta.save(self.filename, med1, med2)

            a = pytta.load(self.filename)

            loadedlst = [a[pyttaobj] for pyttaobj in a]

            for idx, pobj in enumerate(loadedlst):
                self.assertEqual(pobj.lengthDomain,
                                savedlst[idx].lengthDomain)

                self.assertEqual(pobj.excitation.timeSignal.tolist(),
                                savedlst[idx].excitation.timeSignal.tolist())

                self.assertEqual(pobj.device,
                                savedlst[idx].device)
                
                self.assertEqual(str(pobj.inChannels),
                                str(savedlst[idx].inChannels))

                self.assertEqual(str(pobj.outChannels),
                                str(savedlst[idx].outChannels))
                
                self.assertEqual(pobj.samplingRate,
                                savedlst[idx].samplingRate)

                self.assertEqual(pobj.freqMin,
                                savedlst[idx].freqMin)

                self.assertEqual(pobj.freqMax,
                                savedlst[idx].freqMax)
                
                self.assertEqual(pobj.comment,
                                savedlst[idx].comment)

    def test_h5save_analysis(self):
        """
        Analysis hdf5 save test.
        """
        anType = 'RT'
        anType2 = 'mixed'
        nthOct = 3
        nthOct2 = 1
        minBand = 60
        minBand2 = 70
        maxBand = 20000
        maxBand2 = 16000
        data = [2.0226, 1.7139, 1.4615,
               1.7127, 1.0890, 1.5395, 1.2965, 1.9011, 1.9835, 2.1028,
               2.1225, 1.9030, 1.9064, 2.0137, 1.8834, 1.6736, 1.5220,
               1.5677, 1.6691, 1.4698, 1.2754, 0.9378, 0.6863, 0.4889,
               0.3776, 0.3113]
        data2 = [2.0226,
                1.7127, 1.2965, 2.1028,
                1.9064, 1.6736, 1.6691, 0.9378,
                0.3776]
        error = [1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1,
                 1, 1]
        errorLabel='errinho'
        error2 = [1,
                  1, 1, 1,
                  1, 1, 1, 1,
                  1]
        errorLabel2='erroncio'
        dataLabel = 'etiquetismo'  
        dataLabel2 = 'etiquetasso'
        comment = 'Testaaano carai'
        comment2 = 'Testano memo'

        an = pytta.Analysis(anType=anType,
                            nthOct=nthOct,
                            minBand=minBand,
                            maxBand=maxBand,
                            data=data,
                            error=error,
                            errorLabel=errorLabel,
                            dataLabel=dataLabel,
                            comment=comment)

        an2 = pytta.Analysis(anType=anType2,
                            nthOct=nthOct2,
                            minBand=minBand2,
                            maxBand=maxBand2,
                            data=data2,
                            error=error2,
                            errorLabel=errorLabel2,
                            dataLabel=dataLabel2,
                            comment=comment2)

        savedlst = [an, an2]

        pytta.save(self.filename, an, an2)

        a = pytta.load(self.filename)

        loadedlst = [a[pyttaobj] for pyttaobj in a]

        for idx, pobj in enumerate(loadedlst):
            self.assertEqual(pobj.anType,
                            savedlst[idx].anType)

            self.assertEqual(pobj.nthOct,
                            savedlst[idx].nthOct)

            self.assertEqual(pobj.minBand,
                            savedlst[idx].minBand)
            
            self.assertEqual(pobj.maxBand,
                            savedlst[idx].maxBand)

            self.assertEqual(pobj.dataLabel,
                            savedlst[idx].dataLabel)

            self.assertEqual(pobj.errorLabel,
                            savedlst[idx].errorLabel)

            self.assertEqual(pobj.comment,
                            savedlst[idx].comment)

            self.assertEqual(pobj.data.tolist(),
                                savedlst[idx].data.tolist())

            self.assertEqual(pobj.error.tolist(),
                                savedlst[idx].error.tolist())    

if __name__ == '__main__':
    unittest.main()
