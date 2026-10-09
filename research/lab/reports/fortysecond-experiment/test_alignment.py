import unittest
import numpy as np
from common import *
from alignment import ncc_grid,peak,masks
class Tests(unittest.TestCase):
 def test_known_translation_and_noisy_holdout(self):
  rng=np.random.default_rng(42);r=rng.normal(size=(48,48,48));offset=np.array([11,5,9]);t=r[11:43,5:37,9:41].copy();parts=masks();mask=np.logical_or.reduce([x['mask'] for x in parts if x['role']=='fit']);scores=ncc_grid(t,r,mask);p=peak(scores)
  self.assertEqual(p['integer_shift_zyx'],[3,-3,1]);self.assertAlmostEqual(p['NCC'],1,places=12)
  for s in [(0,0,0),(8,8,8),(11,5,9),(16,16,16)]:
   patch=r[s[0]:s[0]+32,s[1]:s[1]+32,s[2]:s[2]+32];a=t[mask];b=patch[mask];expected=np.corrcoef(a,b)[0,1];self.assertAlmostEqual(scores[s],expected,places=12)
 def test_fit_and_check_supports_disjoint_with_gap(self):
  parts=masks();a=np.argwhere(np.logical_or.reduce([p['mask'] for p in parts if p['role']=='fit']));b=np.argwhere(np.logical_or.reduce([p['mask'] for p in parts if p['role']=='check']));distance=np.max(abs(a[:,None]-b[None]),axis=2).min();self.assertGreaterEqual(distance,11)
 def test_flat_template_rejected(self):
  with self.assertRaises(ValueError):ncc_grid(np.zeros((32,)*3),np.ones((48,)*3),np.ones((32,)*3,bool))
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));save(HERE/'ALIGNMENT_TESTS.json',dict(tests_run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors),failed=len(r.failures),errors=len(r.errors),new_model_calls=0));raise SystemExit(not r.wasSuccessful())
