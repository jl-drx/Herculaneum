import unittest
from common import *

class ContractTests(unittest.TestCase):
    def test_midpoint_preserves_half_ulp(self):
        a=np.array([.5],np.float32);b=np.nextafter(a,np.float32(1))
        r=midpoint(a,b)
        self.assertEqual(r.dtype,np.float64)
        self.assertEqual(r[0],.5000000298023224)
        self.assertNotEqual(r[0],float(((a+b)*.5)[0]))
    def test_rejects_nonfinite_and_wrong_dtype(self):
        with self.assertRaises(ValueError):midpoint(np.array([np.nan],np.float32),np.zeros(1,np.float32))
        with self.assertRaises(ValueError):midpoint(np.zeros(1),np.zeros(1))
    def test_exact_ties_and_excluded_support(self):
        metric=metric_module(LAB/'reports/sixth-experiment/benchmark/binary_metrics.py')
        y=np.array([True,False,True,False]);s=np.array([.9,.9,.2,.1]);m=np.array([1,1,1,0],bool)
        r,_=metric.binary_metrics(y[m],s[m])
        self.assertAlmostEqual(r['average_precision'],7/12)
        self.assertAlmostEqual(r['roc_auc'],.25)
    def test_gate_boundary_and_descriptive_separation(self):
        cases=[dict(id=str(i),parent=p) for i,p in enumerate(['Paris1','Paris2','PHerc0814'])]
        roles={r:summarise({c['id']:{k:.25 for k in KEYS} for c in cases},cases) for r in ROLES}
        d=decisions(roles)
        self.assertFalse(d['advancement_screen']['passes']);self.assertTrue(d['descriptive_relation']['passes'])
        roles['midpoint']['contexts']['2']['roc_auc']=.125
        d=decisions(roles);self.assertFalse(d['descriptive_relation']['passes']);self.assertEqual(len(d['advancement_screen']['regional_failures']),1)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    save(HERE/'TEST_RECEIPT.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),created_at_utc=now()))
    raise SystemExit(not result.wasSuccessful())
