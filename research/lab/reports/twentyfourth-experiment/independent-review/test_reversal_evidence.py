"""Intervention identity/state/reconstruction fixtures; generated tensors only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import review_core as c

def deny_network(event,args):
    if event.startswith('socket.'):raise ValueError('network_forbidden')
sys.addaudithook(deny_network)
import torch

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.maths,cls.evidence,cls.helper=c.inherited()
    def arrays(self):
        logits=np.broadcast_to(np.linspace(-3,3,128,dtype=np.float32),(25,128,128)).copy()
        p=self.helper.sigmoid(logits);h=self.helper.hann();s,w,r=self.evidence.reconstruct(p,h)
        return dict(tile_logits=logits,tile_probabilities=p,hann=h,probability_sum=s,weight_sum=w,raw_scores=r,evaluated_mask=w>0)
    def test_reconstruction(self):
        checks=c.verify_arrays(self.arrays(),self.evidence,self.helper);self.assertEqual(checks['raw_scores']['max_absolute'],0)
    def test_wrong_forward_count(self):
        a=self.arrays();a['tile_logits']=a['tile_logits'][:24]
        with self.assertRaises(ValueError):c.verify_arrays(a,self.evidence,self.helper)
    def test_wrong_sigmoid(self):
        a=self.arrays();a['tile_probabilities'][0,0,0]=.5
        with self.assertRaises(ValueError):c.verify_arrays(a,self.evidence,self.helper)
    def test_missing_coverage(self):
        a=self.arrays();a['evaluated_mask'][50,50]=False
        with self.assertRaises(ValueError):c.verify_arrays(a,self.evidence,self.helper)
    def test_no_quantized_output_expansion(self):
        a=self.arrays();a['uint8']=np.zeros((384,384),np.uint8)
        with self.assertRaises(ValueError):c.verify_arrays(a,self.evidence,self.helper)
    def test_correct_model_identity(self):c.compare_model_identity('seed42','seed42',{'sha256':'a'},{'sha256':'a'},'s','s')
    def test_wrong_model(self):
        with self.assertRaises(ValueError):c.compare_model_identity('seed42','native',{}, {},'s','s')
    def test_wrong_checkpoint(self):
        with self.assertRaises(ValueError):c.compare_model_identity('seed42','seed42',{'sha256':'a'},{'sha256':'b'},'s','s')
    def test_wrong_state(self):
        with self.assertRaises(ValueError):c.compare_model_identity('native','native',{}, {},'s','t')
    def test_forward_reference_exact(self):c.compare_forward_reference({'path':'a','sha256':'b'},{'path':'a','sha256':'b'},'native')
    def test_swapped_forward_reference(self):
        with self.assertRaises(ValueError):c.compare_forward_reference({'path':'a'},{'path':'b'},'native')
    def test_input_pairs(self):
        z,y,x=np.indices((17,384,384),dtype=np.int32);a=(1+(17*z+3*y+5*x)%251).astype(np.float32)
        before=c.raw_hash(a);f,r=c.input_pairs(a,self.helper.normalize)
        self.assertEqual(len(f),25);self.assertEqual(len(r),25);self.assertEqual(c.raw_hash(a),before)
        self.assertEqual([v['bbox_roi_xywh'] for v in f],[v['bbox_roi_xywh'] for v in r])
        self.assertTrue(all(x['input_sha256']!=y['input_sha256'] for x,y in zip(f,r,strict=True)))
        a[0,2,2]+=11;changed,_=c.input_pairs(a,self.helper.normalize);self.assertNotEqual(changed[0],f[0])
    def test_wrong_depth_reject(self):
        with self.assertRaises(ValueError):c.input_pairs(np.ones((16,384,384),np.float32),self.helper.normalize)
    def test_delta_sign(self):
        self.assertEqual(c.metric_delta(dict(roc_auc=.25,average_precision=.5),dict(roc_auc=.5,average_precision=.25)),dict(roc_auc=-.25,average_precision=.25))

class SafeStateTests(unittest.TestCase):
    def fixture(self,path):
        names=[f'k{i:03d}' for i in range(508)]
        state={n:torch.tensor([float(i//2)],dtype=torch.float32) for i,n in enumerate(names)}
        schema=dict(tensor_schema=[dict(name=n,shape=[1]) for n in names],expected_factory_alias_groups=[dict(names=names[i:i+2]) for i in range(0,488,2)])
        torch.save(dict(model=state,config=dict(toy=True),step=74999),path)
        return c.pin(path),schema
    def test_safe_toy_state_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            record,schema=self.fixture(Path(td)/'toy.pth');v=c.seed42_value_hashes(record,'model',schema)
            self.assertEqual(len(v['hashes']),508);self.assertEqual(v['step'],74999);self.assertEqual(v['config'],{'toy':True})
    def test_wrong_selected_key(self):
        with tempfile.TemporaryDirectory() as td:
            record,schema=self.fixture(Path(td)/'toy.pth')
            with self.assertRaises(ValueError):c.seed42_value_hashes(record,'state_dict',schema)
    def test_pin_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            record,schema=self.fixture(Path(td)/'toy.pth');record['sha256']='0'*64
            with self.assertRaises(ValueError):c.seed42_value_hashes(record,'model',schema)
    def test_wrong_tensor_shape(self):
        with tempfile.TemporaryDirectory() as td:
            record,schema=self.fixture(Path(td)/'toy.pth');schema['tensor_schema'][0]['shape']=[2]
            with self.assertRaises(ValueError):c.seed42_value_hashes(record,'model',schema)

class PairedSummaryTests(unittest.TestCase):
    def fixture(self):
        contexts={}
        for i,cid in enumerate(c.IDS):
            models={}
            for role,change in [('seed42',.1),('native',-.2)]:
                f=dict(roc_auc=.4+i*.01,average_precision=.3+i*.01)
                r={k:v+change for k,v in f.items()}
                models[role]=dict(forward=f,reverse=r,reverse_minus_forward=c.metric_delta(r,f))
            contexts[cid]=dict(parent_papyrus='Paris1' if i<2 else 'Paris2',models=models)
        return contexts
    def test_equal_context_parent_means(self):
        from verify_reversal_outputs import parent_metrics
        out=parent_metrics(self.fixture(),'Paris1')
        self.assertEqual(out['context_ids'],c.IDS[:2]);self.assertAlmostEqual(out['models']['seed42']['forward']['roc_auc'],.405)
        self.assertAlmostEqual(out['models']['seed42']['reverse_minus_forward']['roc_auc'],.1)
    def test_difference_of_changes_sign(self):
        from verify_reversal_outputs import parent_metrics
        for parent in ('Paris1','Paris2'):
            out=parent_metrics(self.fixture(),parent)
            self.assertAlmostEqual(out['difference_of_changes']['roc_auc'],-.3)
            self.assertAlmostEqual(out['difference_of_changes']['average_precision'],-.3)
    def test_parent_missing_context_rejected(self):
        from verify_reversal_outputs import parent_metrics
        contexts=self.fixture();del contexts[c.IDS[0]]
        with self.assertRaises((ValueError,KeyError)):parent_metrics(contexts,'Paris1')
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'STOPPED',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scientific_array_reads=0,actual_checkpoint_reads=0,model_calls=0)))
    raise SystemExit(0 if result.wasSuccessful() else 2)
