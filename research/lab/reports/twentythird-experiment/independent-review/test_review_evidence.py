"""Saved-output and fingerprint fixtures, without scientific input or model calls."""
import copy
import json
import unittest
import numpy as np
import review_evidence as e
from review_math import Reject

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.h=e.reference_helpers()
    def fixture(self):
        logits=np.broadcast_to(np.linspace(-3,3,128,dtype=np.float32),(25,128,128)).copy()
        p=self.h.sigmoid(logits);w=self.h.hann();sums,weights,raw=e.reconstruct(p,w)
        return dict([('tile-logits',logits),('tile-probabilities',p),('hann',w),('probability-sum',sums),
            ('weight-sum',weights),('raw-scores',raw),('evaluated-mask',weights>0),('uint8',self.h.encode(raw)),
            ('decoded',self.h.decode(self.h.encode(raw)))])
    def test_complete_saved_evidence(self):
        result=e.check_arrays(self.fixture(),self.h);self.assertEqual(result['raw-scores']['max_absolute'],0)
    def test_wrong_logit_shape(self):
        v=self.fixture();v['tile-logits']=v['tile-logits'][:24]
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_sigmoid_mutation(self):
        v=self.fixture();v['tile-probabilities'][0,0,0]=.5
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_bad_hann(self):
        v=self.fixture();v['hann'][0,0]=1
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_missing_evaluated_pixel(self):
        v=self.fixture();v['evaluated-mask'][50,50]=False
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_wrong_quantization(self):
        v=self.fixture();v['uint8'][2,3]+=1
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_wrong_decoding(self):
        v=self.fixture();v['decoded'][2,3]=.5
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_wrong_sum(self):
        v=self.fixture();v['probability-sum'][20,20]+=.01
        with self.assertRaises((Reject,self.h.Reject)):e.check_arrays(v,self.h)
    def test_fingerprint_shape_reject(self):
        with self.assertRaises(Reject):e.normalized_fingerprints(np.ones((16,384,384),np.float32),self.h.normalize)
    def test_empty_patch_reject(self):
        with self.assertRaises(Reject):e.normalized_fingerprints(np.zeros((17,384,384),np.float32),self.h.normalize)
    def test_synthetic_fingerprints_and_ownership(self):
        z,y,x=np.indices((17,384,384),dtype=np.int32)
        a=((z*7+y*3+x*11)%253+1).astype(np.float32);before=e.raw_hash(a)
        original=e.normalized_fingerprints(a,self.h.normalize)
        self.assertEqual(len(original),25);self.assertEqual(e.raw_hash(a),before)
        self.assertEqual([r['bbox_roi_xywh'] for r in original],e.schedule())
        a[0,1,1]+=23;changed=e.normalized_fingerprints(a,self.h.normalize)
        self.assertNotEqual(changed[0]['input_sha256'],original[0]['input_sha256'])
        self.assertEqual(changed[24],original[24])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'STOPPED',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),actual_array_reads=0,model_calls=0)))
    raise SystemExit(0 if result.wasSuccessful() else 2)
