import unittest
import numpy as np
from features import features,encode
class FeatureTests(unittest.TestCase):
    def test_constant(self):
        r=features(np.full((63,384,384),12345,np.uint16))
        for item in r.values():
            self.assertTrue(np.all(item['numerators'][:,0]==12345*item['denominators'][0]));self.assertTrue(np.all(item['numerators'][:,1]==0))
    def test_ramp(self):
        z,y,x=np.indices((63,384,384));raw=(z*7+y*2+x).astype(np.uint16);r=features(raw)
        for k,zs in [('single',np.arange(63)),('triplet',np.arange(1,63,3))]:
            expected=zs[:,None,None]*7+(3*np.arange(32,96)[None,:,None]+1)*2+(3*np.arange(32,96)[None,None,:]+1)
            self.assertTrue(np.array_equal(r[k]['numerators'][:,0]/r[k]['denominators'][0],expected));self.assertTrue(np.all(r[k]['numerators'][:,1]==0))
    def test_impulse_and_depth_dilution(self):
        raw=np.zeros((63,384,384),np.uint16);raw[1,192:195,192:195]=90;r=features(raw)
        self.assertEqual(r['single']['numerators'][1,1,32,32],80*9*90);self.assertEqual(r['single']['numerators'][1,1,32,33],-9*90)
        self.assertEqual(r['single']['numerators'][1,1,32,37],0);self.assertEqual(r['triplet']['numerators'][0,0,32,32]/27,30)
        self.assertEqual(r['triplet']['numerators'][0,1,32,32],80*9*90)
    def test_uint16_maximum_no_overflow(self):
        r=features(np.full((63,384,384),65535,np.uint16))
        for item in r.values():
            self.assertTrue(np.all(item['numerators'][:,0]==65535*item['denominators'][0]));self.assertTrue(np.all(item['numerators'][:,1]==0))
    def test_excluded_margins(self):
        raw=np.zeros((63,384,384),np.uint16);raw[:,:84]=65535;raw[:,300:]=65535;raw[:,:,:84]=65535;raw[:,:,300:]=65535
        self.assertTrue(all(np.all(v['numerators']==0) for v in features(raw).values()))
    def test_monotone_tied_encoding(self):
        for feature,denom in [('intensity',27),('contrast',2187)]:
            n=np.array([0,1,1,2,1000],np.int64)
            for polarity in (1,-1):
                p=encode(n,denom,feature,polarity)
                self.assertEqual(p[1],p[2]);self.assertTrue(np.all(np.diff(p)*polarity>=0));self.assertNotEqual(p[0],p[1])
    def test_input_rejection(self):
        with self.assertRaises(ValueError):features(np.zeros((63,384,384),np.float32))
if __name__=='__main__':unittest.main()
