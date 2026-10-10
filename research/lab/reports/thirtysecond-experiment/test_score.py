import unittest
import numpy as np
from score import independent_rank,fixed_arms,summarize

class Tests(unittest.TestCase):
    def test_ties_and_unknown(self):
        r=independent_rank(np.array([.7,.7,.1,1.]),np.array([1,0,1,2],np.uint8))
        self.assertAlmostEqual(r['ap'],7/12);self.assertEqual(r['auc'],.25)
    def test_window_edges(self):
        p=np.broadcast_to(np.array([0,.1,.2,.6,1.])[:,None,None],(5,128,128))
        b,c=fixed_arms(p,-1);self.assertAlmostEqual(float(b[0,0]),.1);self.assertAlmostEqual(float(c[0,0]),.1)
        b,c=fixed_arms(p,1);self.assertAlmostEqual(float(b[0,0]),.6);self.assertAlmostEqual(float(c[0,0]),.6)
        b,c=fixed_arms(p,0);self.assertAlmostEqual(float(b[0,0]),.2);self.assertAlmostEqual(float(c[0,0]),.3)
    def test_both_practical_margins_required(self):
        contexts=[dict(displacements=[dict(displacement=d,AP_delta=v) for d,v in zip((-1,0,1),(.05,-.02,.05))]) for _ in range(4)]
        r=summarize(contexts);self.assertGreater(r['mean_robust_AP_delta'],.01);self.assertFalse(r['promising_for_replication'])
        contexts=[dict(displacements=[dict(displacement=d,AP_delta=.005) for d in (-1,0,1)]) for _ in range(4)]
        self.assertFalse(summarize(contexts)['promising_for_replication'])
    def test_missing_context_stops(self):
        with self.assertRaises(ValueError):summarize([])

if __name__=='__main__':unittest.main()
