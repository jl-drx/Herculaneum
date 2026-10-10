import unittest
import numpy as np
from depth_reference import infer,texture_energy,boundary_from_energy

class Mechanics(unittest.TestCase):
    def test_known_linear_spatial_gradient_energy(self):
        y,x=np.indices((16,16)); v=np.broadcast_to(11+3*x+4*y,(63,16,16))
        np.testing.assert_array_equal(texture_energy(v),np.full((63,8,8),25.))
    def test_known_step_and_reversal(self):
        e=np.zeros((63,2,2));e[:32]=100
        a=boundary_from_energy(e);b=boundary_from_energy(e[::-1])
        self.assertTrue(a['accepted'].all() and b['accepted'].all())
        np.testing.assert_array_equal(a['depth'],np.full((2,2),32.5))
        np.testing.assert_array_equal(a['depth']+b['depth'],np.full((2,2),64.))
        np.testing.assert_array_equal(a['orientation'],-b['orientation'])
    def test_competing_boundaries_abstain(self):
        e=np.zeros((63,2,2));e[20:42]=100
        self.assertFalse(boundary_from_energy(e)['accepted'].any())
    def test_known_depth_translation(self):
        for split in [29,32,35]:
            e=np.zeros((63,2,2));e[:split]=100
            m=boundary_from_energy(e)
            self.assertTrue(m['accepted'].all())
            np.testing.assert_array_equal(m['depth'],np.full((2,2),split+0.5))
    def test_constant_abstains(self):
        self.assertFalse(infer(np.full((63,16,16),32768.))[1]['accepted'].any())
    def test_positive_affine_intensity(self):
        rng=np.random.Generator(np.random.PCG64(8675309));v=rng.normal(size=(63,20,20));v[:30]*=10
        a=infer(v)[1];b=infer(v*1.25+1000)[1]
        np.testing.assert_array_equal(a['accepted'],b['accepted'])
        np.testing.assert_allclose(a['depth'],b['depth'],equal_nan=True)
        np.testing.assert_allclose(a['contrast'],b['contrast'],rtol=1e-10,atol=1e-10)
    def test_nonfinite_rejected(self):
        v=np.zeros((63,16,16));v[0,0,0]=np.nan
        with self.assertRaises(ValueError):infer(v)
if __name__=='__main__':unittest.main()
