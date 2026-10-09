"""Fabricated tied-score, support and numerical checks; no scientific arrays."""
import json
import unittest
import numpy as np
import review_math as m

def brute(y,s):
    pos=s[y==1];neg=s[y==0]
    auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
    ap=0.
    for value in np.unique(s)[::-1]:
        above=s>=value
        ap+=(np.sum((s==value)&(y==1))/len(pos))*(np.sum(y[above]==1)/np.sum(above))
    return auc,ap

class RankTests(unittest.TestCase):
    def test_known_scores(self):
        r=m.ranking(np.array([0,0,1,1]),np.array([.1,.4,.35,.8]))
        self.assertEqual(r['auc'],.75);self.assertAlmostEqual(r['ap'],5/6)
    def test_constant_ties(self):
        r=m.ranking(np.array([0,0,0,1]),np.ones(4));self.assertEqual(r['auc'],.5);self.assertEqual(r['ap'],.25)
    def test_two_tied_groups(self):
        r=m.ranking(np.array([0,1,0,1]),np.array([.9,.9,.1,.1]));self.assertEqual(r['auc'],.5);self.assertEqual(r['ap'],.5)
    def test_imbalanced_perfect(self):
        y=np.array([0]*99+[1]);r=m.ranking(y,np.arange(100));self.assertEqual(r['auc'],1);self.assertEqual(r['ap'],1)
    def test_imbalanced_reverse(self):
        y=np.array([0]*99+[1]);r=m.ranking(y,-np.arange(100));self.assertEqual(r['auc'],0);self.assertEqual(r['ap'],.01)
    def test_signed_zero_ties(self):
        r=m.ranking(np.array([0,1]),np.array([-0.,0.]));self.assertEqual(r['unique_scores'],1);self.assertEqual(r['auc'],.5)
    def test_pairwise_and_threshold_oracles(self):
        rng=np.random.default_rng(20261009)
        for n in range(2,65):
            y=rng.integers(0,2,n);y[:2]=[0,1];s=rng.integers(0,5,n)
            expected=brute(y,s);r=m.ranking(y,s)
            self.assertAlmostEqual(r['auc'],expected[0],places=14);self.assertAlmostEqual(r['ap'],expected[1],places=14)
    def test_order_and_reverse_invariance(self):
        y=np.array([0,1,1,0,1,0]);s=np.array([0.,0.,1.,2.,3.,3.]);r=m.ranking(y,s)
        permutation=[3,0,5,1,4,2]
        self.assertEqual(m.ranking(y[permutation],s[permutation]),r)
        self.assertAlmostEqual(m.ranking(y,-s)['auc'],1-r['auc'])
    def test_empty_rejection(self):
        with self.assertRaises(m.Reject):m.ranking(np.array([]),np.array([]))
    def test_single_class_rejection(self):
        with self.assertRaises(m.Reject):m.ranking(np.ones(2),np.arange(2))
    def test_nonfinite_rejection(self):
        with self.assertRaises(m.Reject):m.ranking(np.array([0,1]),np.array([0,np.nan]))
    def test_nonbinary_rejection(self):
        with self.assertRaises(m.Reject):m.ranking(np.array([0,2]),np.array([0,1]))
    def test_shape_rejection(self):
        with self.assertRaises(m.Reject):m.ranking(np.array([0,1]),np.array([[0,1]]))

class SupportTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([[0,1],[255,255]],np.uint8);self.s=np.array([[.2,.8],[np.nan,np.nan]],np.float32)
        self.mask=np.array([[True,True],[False,False]]);self.evaluated=np.ones((2,2),bool)
    def test_exact_primary_not_known_or_all(self):
        y,s=m.selected(self.y,self.s,self.mask,self.evaluated);self.assertEqual(y.tolist(),[0,1]);self.assertEqual(len(s),2)
    def test_missing_primary_fails_no_intersection(self):
        self.evaluated[0,1]=False
        with self.assertRaises(m.Reject):m.selected(self.y,self.s,self.mask,self.evaluated)
    def test_nonfinite_primary_fails_no_filter(self):
        self.s[0,1]=np.nan
        with self.assertRaises(m.Reject):m.selected(self.y,self.s,self.mask,self.evaluated)
    def test_empty_primary_rejected(self):
        with self.assertRaises(m.Reject):m.selected(self.y,self.s,np.zeros((2,2),bool),self.evaluated)
    def test_primary_not_boolean_rejected(self):
        with self.assertRaises(m.Reject):m.selected(self.y,self.s,self.mask.astype(np.uint8),self.evaluated)
    def test_wrong_canvas_rejected(self):
        with self.assertRaises(m.Reject):m.selected(self.y,self.s[:1],self.mask,self.evaluated)

class RepresentationTests(unittest.TestCase):
    def test_quantization_endpoints(self):
        r=m.representations(np.array([0,64/255,128/255,192/255,1],np.float32))
        self.assertEqual(r['uint8'].tolist(),[0,64,128,192,255]);self.assertEqual(r['decoded'].tolist(),[0,0,.5,1,1])
    def test_truncation_not_rounding(self):self.assertEqual(int(m.representations(np.array([.5],np.float32))['uint8'][0]),127)
    def test_dtype_rejection(self):
        with self.assertRaises(m.Reject):m.representations(np.array([.5],np.float64))
    def test_range_rejection(self):
        with self.assertRaises(m.Reject):m.representations(np.array([1.01],np.float32))
    def test_quantization_created_ties(self):
        r=m.representations(np.array([.5001,.5002],np.float32));self.assertEqual(len(np.unique(r['uint8'])),1)
        self.assertEqual(m.ranking(np.array([0,1]),r['raw'])['auc'],1);self.assertEqual(m.ranking(np.array([0,1]),r['uint8'])['auc'],.5)
    def test_decoded_clipping_created_ties(self):
        r=m.representations(np.array([.8,.9],np.float32));self.assertEqual(r['decoded'].tolist(),[1,1])
    def test_zero_one_fractions(self):
        s=np.array([0,.5,1,1],np.float32);r=m.model_metrics(np.array([0,0,1,1]),s,np.ones(4,bool),np.ones(4,bool))
        self.assertEqual(r['decoded_zero_fraction'],.25);self.assertEqual(r['decoded_one_fraction'],.5)
    def test_paired_difference_linear_quantile(self):
        a=np.array([0,.25,.75,1],np.float32);b=np.array([.25,.25,.5,.5],np.float32)
        r=m.disagreement(a,b,np.ones(4,bool));self.assertEqual(r['mean_signed'],.125);self.assertEqual(r['mean_absolute'],.25)
        self.assertAlmostEqual(r['p95_absolute'],.4625);self.assertEqual(r['max_absolute'],.5)
    def test_paired_order_sign(self):
        a=np.array([.1,.8],np.float32);b=np.array([.2,.4],np.float32);mask=np.ones(2,bool)
        ab=m.disagreement(a,b,mask);ba=m.disagreement(b,a,mask)
        self.assertEqual(ab['mean_signed'],-ba['mean_signed']);self.assertEqual(ab['mean_absolute'],ba['mean_absolute'])
    def test_comparison_missing_field(self):
        with self.assertRaises(m.Reject):m.scalar_close({},dict(auc=.5))
    def test_comparison_changed_metric(self):
        with self.assertRaises(m.Reject):m.scalar_close(dict(auc=.5001),dict(auc=.5))
    def test_comparison_tolerance(self):m.scalar_close(dict(auc=.5+1e-13),dict(auc=.5))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'STOPPED',tests=result.testsRun,
        pairwise_oracle_sizes=63,errors=len(result.errors),failures=len(result.failures),actual_array_reads=0,model_calls=0)))
    raise SystemExit(0 if result.wasSuccessful() else 2)
