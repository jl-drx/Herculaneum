import unittest
import numpy as np
from geometry_math import describe
class GeometryTests(unittest.TestCase):
    def plane(self):
        q=np.zeros((3,3,6),np.float64)
        for y in range(3):
            for x in range(3):q[y,x]=[100+2*x+y,200+3*y,1000,0,0,1]
        return q
    def test_affine_plane(self):
        r=describe(self.plane());self.assertEqual(r['tangent_x'],[2,0,0]);self.assertEqual(r['tangent_y'],[1,3,0]);self.assertEqual(r['area_scale'],6)
        self.assertEqual(r['normal_dot_unit_raster_cross'],1);self.assertEqual(r['valid_layer_count'],63)
    def test_sentinel_retained(self):
        q=self.plane();q[0,0]=0;r=describe(q);self.assertEqual(r['mapped_records'],8);self.assertFalse(r['complete']);self.assertIsNone(r['tangent_x'])
    def test_invalid_normal(self):
        q=self.plane();q[1,1,5]=2
        with self.assertRaises(ValueError):describe(q)
    def test_ray_bounds(self):
        q=self.plane();q[:,:,2]=0;r=describe(q);self.assertEqual(r['valid_layer_count'],32);self.assertEqual(r['layer_validity'][:31],[False]*31)
if __name__=='__main__':unittest.main()
