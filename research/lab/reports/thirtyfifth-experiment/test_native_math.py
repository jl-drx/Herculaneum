import unittest
import numpy as np
from native_math import evaluate,CORNERS
class NativeTests(unittest.TestCase):
    def test_asymmetric_historical_corner(self):
        c=np.zeros((1,8));c[0,1]=100;c[0,3]=300;r=evaluate(c,np.array([[.25,.5,.75]]));self.assertEqual(r['standard'][0],12.5);self.assertEqual(r['historical'][0],6.25);self.assertEqual(r['rounded_historical'][0],6)
    def test_constant(self):
        r=evaluate(np.full((1,8),12345),np.array([[.25,.5,.75]]));self.assertEqual(r['standard'][0],12345);self.assertEqual(r['historical'][0],12345)
    def test_correct_ramp(self):
        c=np.array([[100+2*x+3*y+5*z for x,y,z in CORNERS]]);r=evaluate(c,np.array([[.25,.5,.75]]));self.assertEqual(r['standard'][0],105.75)
    def test_nearest_even_ties(self):
        r=evaluate(np.array([[2]*8,[3]*8],float)+.5,np.zeros((2,3)));self.assertEqual(r['rounded_historical'].tolist(),[2,4])
if __name__=='__main__':unittest.main()
