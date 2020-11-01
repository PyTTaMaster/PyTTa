import unittest
import pytta


class TestAnalysisOperations(unittest.TestCase):

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
        self.nthOct = 3
        self.minFreq = 100
        self.maxFreq = 160
        self.a = pytta.Analysis(anType='RT', nthOct=self.nthOct,
                                minBand=self.minFreq,
                                maxBand=self.maxFreq,
                                data=[2, 1, -1])
        self.b = pytta.Analysis(anType='RT', nthOct=self.nthOct,
                                minBand=self.minFreq,
                                maxBand=self.maxFreq,
                                data=[3, 1, -2])

    def tearDown(self):
        """
        It runs after each test
        """
        pass

    def test_add(self):
        test = self.a+self.b
        self.assertEqual(test.data.tolist(), [5,2,-3])
        test = self.b+self.a
        self.assertEqual(test.data.tolist(), [5,2,-3])

    def test_sub(self):
        test = self.a-self.b
        self.assertEqual(test.data.tolist(), [-1,0,1])
        test = self.b-self.a
        self.assertEqual(test.data.tolist(), [1,0,-1])


    def test_mul(self):
        test = self.a*self.b
        self.assertEqual(test.data.tolist(), [6,1,2])
        test = self.b*self.a
        self.assertEqual(test.data.tolist(), [6,1,2])

    def test_div(self):
        test = self.a/self.b
        self.assertEqual(test.data.tolist(), [2/3,1,1/2])
        test = self.b/self.a
        self.assertEqual(test.data.tolist(), [3/2,1,2])


if __name__ == '__main__':
    unittest.main()